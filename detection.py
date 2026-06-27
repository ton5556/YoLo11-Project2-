# region imports
# Standard library imports
import os
os.environ["GST_PLUGIN_FEATURE_RANK"] = "vaapidecodebin:NONE"

import sys
import threading
import queue as queue_module
from pathlib import Path

# Third-party imports
import gi

gi.require_version("Gst", "1.0")
import cv2
import numpy as np

# Local application-specific imports
import hailo
from gi.repository import Gst

from hailo_apps.python.pipeline_apps.detection.detection_pipeline import GStreamerDetectionApp
from hailo_apps.python.core.common.buffer_utils import (
    get_caps_from_pad,
    get_numpy_from_buffer,
)
from hailo_apps.python.core.common.core import get_pipeline_parser
from hailo_apps.python.core.common.hailo_logger import get_logger
from hailo_apps.python.core.gstreamer.gstreamer_app import app_callback_class

# Lane detection imports (conditional — only needed when --lane-hef-path is used)
LANE_HAILO_AVAILABLE = False
try:
    _lane_det_dir = str(
        Path(__file__).resolve().parents[2] / "standalone_apps" / "lane_detection"
    )
    if _lane_det_dir not in sys.path:
        sys.path.insert(0, _lane_det_dir)
    from lane_detection_utils import UFLDProcessing, compute_scaled_radius
    from hailo_apps.python.core.common.hailo_inference import HailoInfer

    LANE_HAILO_AVAILABLE = True
    print("[LANE] ✅ Hailo UFLD imports OK")
except ImportError as e:
    # *** THIS WAS SILENTLY SWALLOWED BEFORE — now visible ***
    print(f"[LANE] ⚠ Hailo lane imports failed: {e}")
    print("[LANE]   → Will use OpenCV lane detection as fallback if --lane-hef-path is given.")

hailo_logger = get_logger(__name__)
# endregion imports


# ===============================================================================================
# Lane Detection — draw utility (shared by both backends)
# ===============================================================================================

def draw_lanes_on_frame(frame, lanes, radius=5):
    """
    Draw lane detection results directly on a frame (in-place).

    Draws colored points + connecting lines for each detected lane,
    and a semi-transparent green fill between the two inner lanes.

    Args:
        frame: numpy array (H, W, 3) — RGB or BGR (colors chosen to be symmetric).
        lanes: list of lanes, each lane is a list of (x, y) tuples.
        radius: circle radius for lane points.
    """
    lane_colors = [
        (0, 255, 0),    # Green  (same in RGB & BGR)
        (0, 255, 200),  # Cyan-green
        (255, 255, 0),  # Yellow
        (200, 100, 0),  # Orange-ish
    ]

    for i, lane in enumerate(lanes):
        color = lane_colors[i % len(lane_colors)]
        for coord in lane:
            cv2.circle(frame, coord, radius, color, -1)
        if len(lane) > 1:
            for j in range(len(lane) - 1):
                cv2.line(frame, lane[j], lane[j + 1], color, 2)

    # Semi-transparent fill between the first two lanes (drivable area)
    if len(lanes) >= 2 and len(lanes[0]) > 2 and len(lanes[1]) > 2:
        try:
            pts = np.array(lanes[0] + lanes[1][::-1], dtype=np.int32)
            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts], (0, 255, 0))
            cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
        except Exception:
            pass


# ===============================================================================================
# Lane Detection Backend A — Hailo UFLD model (accurate, needs hailo_platform)
# ===============================================================================================

class LaneDetectorHailo:
    """
    Runs UFLD v2 lane detection in a background thread using HailoRT (HailoInfer).
    Requires: hailo_platform (PyHailoRT), ufld_v2_tu.hef
    """

    def __init__(self, hef_path, frame_width=1280, frame_height=720):
        print(f"[LANE-Hailo] Initializing with HEF: {hef_path}")

        self.frame_width = frame_width
        self.frame_height = frame_height

        # UFLD v2 model parameters (fixed for ufld_v2_tu)
        self.ufld = UFLDProcessing(
            num_cell_row=100,
            num_cell_col=100,
            num_row=56,
            num_col=41,
            num_lanes=4,
            crop_ratio=0.8,
            original_frame_width=frame_width,
            original_frame_height=frame_height,
            total_frames=0,
        )

        # HailoRT inference engine
        self.hailo_infer = HailoInfer(hef_path, batch_size=1, output_type="FLOAT32")
        self.input_height, self.input_width, _ = self.hailo_infer.get_input_shape()
        print(f"[LANE-Hailo] Model input shape: {self.input_width}x{self.input_height}")

        self.radius = compute_scaled_radius(frame_width, frame_height)

        # Thread-safe lane storage
        self._latest_lanes = []
        self._lanes_lock = threading.Lock()
        self._input_queue = queue_module.Queue(maxsize=2)
        self._running = True
        self._inference_count = 0

        self._thread = threading.Thread(
            target=self._inference_loop, daemon=True, name="LaneDetectorHailo"
        )
        self._thread.start()
        print("[LANE-Hailo] ✅ Background inference thread started.")

    def _inference_loop(self):
        """Background inference loop for UFLD lane detection."""
        while self._running:
            try:
                frame_rgb = self._input_queue.get(timeout=1.0)
            except queue_module.Empty:
                continue

            try:
                # Convert RGB → BGR (UFLD model expects BGR from OpenCV pipeline)
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                resized = self.ufld.resize(frame_bgr, self.input_height, self.input_width)

                # Synchronous-style inference via async + wait
                result_holder = [None]

                def on_complete(completion_info, bindings_list):
                    if completion_info.exception:
                        hailo_logger.error("Lane inference error: %s", completion_info.exception)
                        return
                    for bindings in bindings_list:
                        if len(bindings._output_names) == 1:
                            result_holder[0] = bindings.output().get_buffer()
                        else:
                            result_holder[0] = {
                                name: np.expand_dims(
                                    bindings.output(name).get_buffer(), axis=0
                                )
                                for name in bindings._output_names
                            }

                job = self.hailo_infer.run([resized], on_complete)
                job.wait(10000)

                if result_holder[0] is not None:
                    result = result_holder[0]
                    if isinstance(result, dict):
                        slices = list(result.values())
                        output_tensor = np.concatenate(slices, axis=1)
                    else:
                        output_tensor = result
                        if output_tensor.ndim == 1:
                            output_tensor = np.expand_dims(output_tensor, axis=0)

                    lanes = self.ufld.get_coordinates(output_tensor)
                    with self._lanes_lock:
                        self._latest_lanes = lanes

                    self._inference_count += 1
                    if self._inference_count <= 3 or self._inference_count % 100 == 0:
                        print(f"[LANE-Hailo] Inference #{self._inference_count}: found {len(lanes)} lanes")

            except Exception as e:
                hailo_logger.error("Lane-Hailo inference error: %s", e)
                print(f"[LANE-Hailo] ❌ Inference error: {e}")

    def submit_frame(self, frame_rgb):
        """Submit an RGB frame for lane detection (non-blocking)."""
        while not self._input_queue.empty():
            try:
                self._input_queue.get_nowait()
            except queue_module.Empty:
                break
        try:
            self._input_queue.put_nowait(frame_rgb.copy())
        except queue_module.Full:
            pass

    def get_latest_lanes(self):
        """Return the most recent lane coordinates (thread-safe)."""
        with self._lanes_lock:
            return list(self._latest_lanes)

    def stop(self):
        """Stop the background thread and release Hailo resources."""
        self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=5.0)
        self.hailo_infer.close()
        print("[LANE-Hailo] Stopped.")


# ===============================================================================================
# Lane Detection Backend B — OpenCV Classical (fallback, no Hailo needed)
# ===============================================================================================

class LaneDetectorCV:
    """
    Classical lane detection using Canny edge detection + Hough lines.
    Runs on CPU, no Hailo chip needed. Less accurate than UFLD but always works.
    """

    def __init__(self, frame_width=1280, frame_height=720):
        print(f"[LANE-CV] Initializing OpenCV lane detection ({frame_width}x{frame_height})")

        self.frame_width = frame_width
        self.frame_height = frame_height
        self.radius = 5

        self._latest_lanes = []
        self._lanes_lock = threading.Lock()
        self._input_queue = queue_module.Queue(maxsize=2)
        self._running = True
        self._detect_count = 0

        self._thread = threading.Thread(
            target=self._detect_loop, daemon=True, name="LaneDetectorCV"
        )
        self._thread.start()
        print("[LANE-CV] ✅ Background detection thread started.")

    def _detect_loop(self):
        """Background loop for OpenCV-based lane detection."""
        while self._running:
            try:
                frame = self._input_queue.get(timeout=1.0)
            except queue_module.Empty:
                continue

            try:
                lanes = self._detect_lanes(frame)
                with self._lanes_lock:
                    self._latest_lanes = lanes

                self._detect_count += 1
                if self._detect_count <= 3 or self._detect_count % 100 == 0:
                    print(f"[LANE-CV] Detection #{self._detect_count}: found {len(lanes)} lanes")

            except Exception as e:
                hailo_logger.error("Lane-CV detection error: %s", e)

    def _detect_lanes(self, frame_rgb):
        """Detect lane lines using Canny + Hough transform."""
        h, w = frame_rgb.shape[:2]

        # Convert to grayscale
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)

        # Blur to reduce noise
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Canny edge detection
        edges = cv2.Canny(blur, 50, 150)

        # Region of interest — bottom-half trapezoid (road area)
        mask = np.zeros_like(edges)
        roi_vertices = np.array([[
            (int(w * 0.05), h),
            (int(w * 0.40), int(h * 0.55)),
            (int(w * 0.60), int(h * 0.55)),
            (int(w * 0.95), h),
        ]], dtype=np.int32)
        cv2.fillPoly(mask, roi_vertices, 255)
        masked_edges = cv2.bitwise_and(edges, mask)

        # Hough line transform
        lines = cv2.HoughLinesP(
            masked_edges, rho=1, theta=np.pi / 180, threshold=50,
            minLineLength=40, maxLineGap=150,
        )

        if lines is None:
            return []

        # Separate left and right lane lines by slope
        left_lines = []
        right_lines = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 == x1:
                continue
            slope = (y2 - y1) / (x2 - x1)
            if abs(slope) < 0.3:  # Filter nearly horizontal lines
                continue
            if slope < 0:
                left_lines.append((x1, y1, x2, y2))
            else:
                right_lines.append((x1, y1, x2, y2))

        lanes = []

        for group in [left_lines, right_lines]:
            if not group:
                continue

            # Average line parameters
            x1s = np.mean([l[0] for l in group])
            y1s = np.mean([l[1] for l in group])
            x2s = np.mean([l[2] for l in group])
            y2s = np.mean([l[3] for l in group])

            dx = x2s - x1s
            if abs(dx) < 1:
                continue
            slope = (y2s - y1s) / dx
            intercept = y1s - slope * x1s

            # Extend lane line from bottom to 60% height
            y_bottom = h
            y_top = int(h * 0.6)
            x_bottom = int((y_bottom - intercept) / slope)
            x_top = int((y_top - intercept) / slope)

            # Clamp x coordinates
            x_bottom = max(0, min(w - 1, x_bottom))
            x_top = max(0, min(w - 1, x_top))

            # Create lane as list of points
            num_points = 10
            lane = []
            for t in range(num_points):
                frac = t / (num_points - 1)
                x = int(x_top + frac * (x_bottom - x_top))
                y = int(y_top + frac * (y_bottom - y_top))
                lane.append((x, y))
            if lane:
                lanes.append(lane)

        return lanes

    def submit_frame(self, frame_rgb):
        """Submit an RGB frame for lane detection (non-blocking)."""
        while not self._input_queue.empty():
            try:
                self._input_queue.get_nowait()
            except queue_module.Empty:
                break
        try:
            self._input_queue.put_nowait(frame_rgb.copy())
        except queue_module.Full:
            pass

    def get_latest_lanes(self):
        """Return the most recent lane coordinates (thread-safe)."""
        with self._lanes_lock:
            return list(self._latest_lanes)

    def stop(self):
        """Stop the background thread."""
        self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=5.0)
        print("[LANE-CV] Stopped.")


# -----------------------------------------------------------------------------------------------
# User-defined class to be used in the callback function
# -----------------------------------------------------------------------------------------------
class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()
        self.new_variable = 42
        # Lane detection (set by main() when --lane-hef-path is provided)
        self.lane_detector = None
        self.lane_frame_interval = 2  # Run lane inference every N frames

    def new_function(self):
        return "The meaning of life is: "


# -----------------------------------------------------------------------------------------------
# User-defined callback function
# -----------------------------------------------------------------------------------------------


def app_callback(element, buffer, user_data):
    if buffer is None:
        hailo_logger.warning("Received None buffer.")
        return

    # Note: Frame counting is handled automatically by the framework wrapper
    frame_idx = user_data.get_count()
    string_to_print = f"Frame count: {user_data.get_count()}\n"

    pad = element.get_static_pad("src")
    format, width, height = get_caps_from_pad(pad)

    # ----- Lane Detection: draw on GStreamer buffer directly -----
    lanes_drawn_on_buffer = False
    if (
        user_data.lane_detector is not None
        and format is not None
        and width is not None
        and height is not None
    ):
        # Submit frame for lane inference every N frames
        if frame_idx % user_data.lane_frame_interval == 0:
            lane_input = get_numpy_from_buffer(buffer, format, width, height)
            if lane_input is not None:
                user_data.lane_detector.submit_frame(lane_input)

        # Draw latest lane coordinates on the GStreamer buffer (every frame)
        lanes = user_data.lane_detector.get_latest_lanes()
        if lanes:
            try:
                success, map_info = buffer.map(
                    Gst.MapFlags.READ | Gst.MapFlags.WRITE
                )
                if success:
                    try:
                        if format == "RGB":
                            # numpy view of the buffer memory (no copy — writes go to buffer)
                            frame_view = np.ndarray(
                                shape=(height, width, 3),
                                dtype=np.uint8,
                                buffer=map_info.data,
                            )
                            draw_lanes_on_frame(
                                frame_view,
                                lanes,
                                user_data.lane_detector.radius,
                            )
                            lanes_drawn_on_buffer = True
                    finally:
                        buffer.unmap(map_info)
                else:
                    # First failure: print a visible warning
                    if frame_idx <= 5:
                        print("[LANE] ⚠ Buffer is not writable — lane overlay will appear only in --use-frame window")
            except Exception as e:
                if frame_idx <= 5:
                    print(f"[LANE] ⚠ Buffer map error: {e}")

    # ----- Object Detection (existing logic, unchanged) -----
    frame = None
    if user_data.use_frame and format is not None and width is not None and height is not None:
        # If lanes were drawn on the buffer above, this copy includes them
        frame = get_numpy_from_buffer(buffer, format, width, height)

    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    detection_count = 0
    for detection in detections:
        label = detection.get_label()
        confidence = detection.get_confidence()
        if label == "person":
            # Get track ID
            track_id = 0
            track = detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)
            if len(track) == 1:
                track_id = track[0].get_id()
            string_to_print += (
                f"Detection: ID: {track_id} Label: {label} Confidence: {confidence:.2f}\n"
            )
            detection_count += 1
    if user_data.use_frame:
        # If lanes were NOT drawn on the GStreamer buffer, draw on cv2 frame as fallback
        if (
            not lanes_drawn_on_buffer
            and user_data.lane_detector is not None
        ):
            lanes = user_data.lane_detector.get_latest_lanes()
            if lanes:
                draw_lanes_on_frame(
                    frame, lanes, user_data.lane_detector.radius
                )

        cv2.putText(
            frame,
            f"Detections: {detection_count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            f"{user_data.new_function()} {user_data.new_variable}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        user_data.set_frame(frame)

    print(string_to_print)
    return


def main():
    hailo_logger.info("Starting Detection App.")

    # Create parser and add lane detection arguments
    parser = get_pipeline_parser()
    parser.add_argument(
        "--lane-hef-path",
        default=None,
        help=(
            "Path to the lane detection HEF file (e.g., ufld_v2_tu.hef). "
            "Enables lane overlay alongside object detection. "
            "If Hailo UFLD is unavailable, falls back to OpenCV lane detection."
        ),
    )
    parser.add_argument(
        "--lane-interval",
        type=int,
        default=2,
        help="Run lane detection every N frames (default: 2).",
    )

    user_data = user_app_callback_class()
    app = GStreamerDetectionApp(app_callback, user_data, parser=parser)

    # Read parsed lane arguments
    lane_hef_path = getattr(app.options_menu, "lane_hef_path", None)
    lane_interval = getattr(app.options_menu, "lane_interval", 2)

    # Initialize lane detector if requested
    if lane_hef_path is not None:
        user_data.lane_frame_interval = lane_interval
        print(f"\n{'='*60}")
        print(f"[LANE] Lane detection requested: {lane_hef_path}")
        print(f"[LANE] Interval: every {lane_interval} frames")
        print(f"[LANE] Hailo UFLD available: {LANE_HAILO_AVAILABLE}")
        print(f"{'='*60}")

        # Try Hailo UFLD first, fall back to OpenCV
        if LANE_HAILO_AVAILABLE:
            try:
                user_data.lane_detector = LaneDetectorHailo(
                    hef_path=lane_hef_path,
                    frame_width=app.video_width,
                    frame_height=app.video_height,
                )
                print("[LANE] ✅ Using Hailo UFLD model for lane detection")
            except Exception as e:
                print(f"[LANE] ❌ Hailo UFLD init failed: {e}")
                print("[LANE] ↓ Falling back to OpenCV lane detection...")
                user_data.lane_detector = LaneDetectorCV(
                    frame_width=app.video_width,
                    frame_height=app.video_height,
                )
                print("[LANE] ✅ Using OpenCV lane detection (fallback)")
        else:
            print("[LANE] ⚠ hailo_platform / PyHailoRT not installed")
            print("[LANE] ↓ Using OpenCV lane detection...")
            user_data.lane_detector = LaneDetectorCV(
                frame_width=app.video_width,
                frame_height=app.video_height,
            )
            print("[LANE] ✅ Using OpenCV lane detection (fallback)")

        print(f"{'='*60}\n")

    app.run()


if __name__ == "__main__":
    main()
