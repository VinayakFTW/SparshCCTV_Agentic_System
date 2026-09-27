import os
import time
from typing import Optional, Tuple, Union
import cv2
import numpy as np


class StreamCapture:
    """Robust video capture wrapper for RTSP CCTV streams and video files."""

    def __init__(self, source: Union[str, int], name: Optional[str] = None, target_fps: Optional[float] = None):
        self.source = source
        self.target_fps = target_fps
        self.frame_interval = (1.0 / target_fps) if target_fps and target_fps > 0 else 0.0
        self.last_frame_time = 0.0
        self.is_rtsp = isinstance(source, str) and (
            source.startswith("rtsp://") or source.startswith("rtsps://")
        )
        self.is_live = self.is_rtsp or isinstance(source, int)
        self.name = name or (f"Sparsh CCTV ({source})" if self.is_rtsp else str(source))

        if self.is_rtsp:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                "rtsp_transport;udp|fflags;nobuffer|flags;low_delay"
            )

        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Failed to open video source: {self.source}\n"
                f"For RTSP: verify network connection, IP, credentials, and camera power.\n"
                f"For files: verify the file path exists and is readable."
            )

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0 or np.isnan(self.fps):
            self.fps = 15.0 if target_fps else 30.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not self.is_live else -1
        self.frame_idx = 0

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Reads the next video frame with optional 15 FPS clock pacing."""
        if self.frame_interval > 0:
            elapsed = time.time() - self.last_frame_time
            if elapsed < self.frame_interval:
                time.sleep(self.frame_interval - elapsed)
            self.last_frame_time = time.time()

        ret, frame = self.cap.read()
        if ret:
            self.frame_idx += 1
            return True, frame
        
        # If live stream, attempt a single reconnect attempt
        if self.is_rtsp:
            print("[StreamCapture] RTSP feed interrupted, attempting reconnection...")
            self.cap.release()
            time.sleep(1.0)
            self.cap = cv2.VideoCapture(self.source)
            ret, frame = self.cap.read()
            if ret:
                self.frame_idx += 1
                return True, frame

        return False, None

    def release(self):
        """Release underlying OpenCV VideoCapture."""
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
