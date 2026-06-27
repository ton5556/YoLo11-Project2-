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
LANE_DETECTION_AVAILABLE = False
try:
    _lane_det_dir = str(
        Path(__file__).resolve().parents[2] / "standalone_apps" / "lane_detection"
    )
    if _lane_det_dir not in sys.path:
        sys.path.insert(0, _lane_det_dir)
    from lane_detection_utils import UFLDProcessing, compute_scaled_radius
    from hailo_apps.python.core.common.hailo_inference import HailoInfer

    LANE_DETECTION_AVAILABLE = True
except ImportError:
    pass

hailo_logger = get_logger(__name__)
# endregion imports


# -----------------------------------------------------------------------------------------------
# Lane Detection Background Processor
# -----------------------------------------------------------------------------------------------
class LaneDetector:
    """
    Runs UFLD v2 lane detection in a background thread using HailoRT (HailoInfer).

    This class manages its own HailoRT VDevice with group_id="SHARED", allowing it
    to coexist with the GStreamer hailonet element on the same Hailo chip via the
    built-in round-robin scheduler.

    Usage:
        detector = LaneDetector("path/to/ufld_v2_tu.hef", 1280, 720)
        detector.submit_frame(rgb_frame)       # non-blocking, drops old frames
        lanes = detector.get_latest_lanes()    # thread-safe read
        LaneDetector.draw_lanes(frame, lanes)  # static drawing utility
        detector.stop()                        # cleanup
    """

    def __init__(self, hef_path, frame_width=1280, frame_height=720):
        hailo_logger.info("Initializing Lane Detector with HEF: %s", hef_path)

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
        hailo_logger.info(
            "Lane model input shape: %dx%d", self.input_width, self.input_height
        )

        # Scaled drawing radius for lane points
        self.radius = compute_scaled_radius(frame_width, frame_height)

        # Thread-safe storage for latest lane coordinates
        self._latest_lanes = []
        self._lanes_lock = threading.Lock()

        # Input queue — maxsize=2 so we always process near-latest frames
        self._input_queue = queue_module.Queue(maxsize=2)
        self._running = True

        # Background inference thread
        self._thread = threading.Thread(
            target=self._inference_loop, daemon=True, name="LaneDetector"
        )
        self._thread.start()
        hailo_logger.info("Lane Detector started.")

    # ---- Background inference loop ----

    def _inference_loop(self):
        """Continuously pull frames from the queue and run lane detection."""
        while self._running:
            try:
                frame_rgb = self._input_queue.get(timeout=1.0)
            except queue_module.Empty:
                continue

            try:
                # Convert RGB → BGR (model was trained/compiled with BGR input)
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

                # Preprocess: resize + crop for UFLD model
                resized = self.ufld.resize(
                    frame_bgr, self.input_height, self.input_width
                )

                # Synchronous-style inference via async + wait
                result_holder = [None]

                def on_complete(completion_info, bindings_list):
                    if completion_info.exception:
                        hailo_logger.error(
                            "Lane inference error: %s", completion_info.exception
                        )
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

            except Exception as e:
                hailo_logger.error("Lane detection error: %s", e)

    # ---- Public API ----

    def submit_frame(self, frame_rgb):
        """
        Submit an RGB frame for lane detection (non-blocking).
        Old unprocessed frames are dropped to keep latency low.
        """
        # Flush stale frames
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
        hailo_logger.info("Stopping Lane Detector...")
        self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=5.0)
        self.hailo_infer.close()
        hailo_logger.info("Lane Detector stopped.")

    # ---- Drawing utility (static — usable without an instance) ----

    @staticmethod
    def draw_lanes(frame, lanes, radius=5):
        """
        Draw lane detection results directly on a frame (in-place).

        Draws colored points + connecting lines for each detected lane,
        and a semi-transparent green fill between the two inner lanes.

        Args:
            frame: numpy array (H, W, 3) in RGB or BGR — colors are symmetric.
            lanes: list of lanes, each lane is a list of (x, y) tuples.
            radius: circle radius for lane points.
        """
        # Lane colors — green is (0,255,0) in both RGB and BGR
        lane_colors = [
            (0, 255, 0),  # Green
            (0, 255, 200),  # Cyan-green
            (255, 255, 0),  # Yellow
            (200, 100, 0),  # Orange-ish
        ]

        for i, lane in enumerate(lanes):
            color = lane_colors[i % len(lane_colors)]
            # Draw points
            for coord in lane:
                cv2.circle(frame, coord, radius, color, -1)
            # Connect consecutive points with lines
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
                pass  # Skip transparency if it fails on this buffer type


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
                            LaneDetector.draw_lanes(
                                frame_view,
                                lanes,
                                user_data.lane_detector.radius,
                            )
                            lanes_drawn_on_buffer = True
                    finally:
                        buffer.unmap(map_info)
            except Exception as e:
                hailo_logger.debug("Cannot draw lanes on buffer: %s", e)

    # ----- Object Detection (existing logic, unchanged) -----
    frame = None
    if user_data.use_frame and format is not None and width is not None and height is not None:
        # This read-maps the buffer and copies — if lanes were drawn above,
        # this copy will already include the lane overlay.
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
        # If lanes were NOT drawn on the GStreamer buffer, draw on cv2 frame instead
        if (
            not lanes_drawn_on_buffer
            and user_data.lane_detector is not None
        ):
            lanes = user_data.lane_detector.get_latest_lanes()
            if lanes:
                LaneDetector.draw_lanes(
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

    # Create parser and add --lane-hef-path before GStreamerDetectionApp parses it
    parser = get_pipeline_parser()
    parser.add_argument(
        "--lane-hef-path",
        default=None,
        help=(
            "Path to the lane detection HEF file (e.g., ufld_v2_tu.hef). "
            "When provided, lane detection overlay is drawn on the display "
            "alongside object detection boxes. Omit to run detection only."
        ),
    )
    parser.add_argument(
        "--lane-interval",
        type=int,
        default=2,
        help=(
            "Run lane detection every N frames (default: 2). "
            "Higher values reduce Hailo chip load but make lane lines less responsive."
        ),
    )

    user_data = user_app_callback_class()
    app = GStreamerDetectionApp(app_callback, user_data, parser=parser)

    # Read parsed lane arguments
    lane_hef_path = getattr(app.options_menu, "lane_hef_path", None)
    lane_interval = getattr(app.options_menu, "lane_interval", 2)

    # Initialize lane detector if requested
    if lane_hef_path is not None:
        if not LANE_DETECTION_AVAILABLE:
            hailo_logger.error(
                "Lane detection dependencies not available. "
                "Ensure lane_detection_utils.py and HailoInfer are accessible."
            )
        else:
            user_data.lane_frame_interval = lane_interval
            user_data.lane_detector = LaneDetector(
                hef_path=lane_hef_path,
                frame_width=app.video_width,
                frame_height=app.video_height,
            )
            hailo_logger.info(
                "Lane detection enabled (interval=%d frames, hef=%s)",
                lane_interval,
                lane_hef_path,
            )

    app.run()


if __name__ == "__main__":
    main()
