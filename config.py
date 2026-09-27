import os
from dataclasses import dataclass, field
from typing import Dict, Tuple

# Base Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
SNAPSHOT_DIR = os.path.join(OUTPUT_DIR, "snapshots")
RECORDING_DIR = os.path.join(OUTPUT_DIR, "recordings")

os.makedirs(SNAPSHOT_DIR, exist_ok=True)
os.makedirs(RECORDING_DIR, exist_ok=True)

# Default Model Paths
CRASH_MODEL_DIR = os.path.join(MODELS_DIR, "crash_classifier")
CRASH_MODEL_PATH = os.path.join(CRASH_MODEL_DIR, "model.safetensors")
FIRE_MODEL_PATH = os.path.join(MODELS_DIR, "fire_classifier", "fire_classifier.pt")

# YOLO Hierarchical Accident & Severity Detector
ACCIDENT_DETECTOR_BACKEND = "yolo"  # "yolo" or "detr"
YOLO_DETECTOR_PATH = os.path.join(MODELS_DIR, "accident_classifier", "yolo_detector.pt")
YOLO_SEVERITY_PATH = os.path.join(MODELS_DIR, "accident_classifier", "yolo_severity.pt")

# Camera Stream & Server FPS Configuration
DEFAULT_CAMERA_FPS = 15.0  # Camera stream runs on 15 FPS cap
FRAME_TIME_MS = 1000.0 / DEFAULT_CAMERA_FPS  # 66.66 ms budget per frame

# Default Video & Camera Settings
DEFAULT_TEST_VIDEO = os.path.join(BASE_DIR, "data", "videos", "v3.mp4")

# Sparsh CCTV RTSP Configuration
RTSP_USERNAME = "admin"
RTSP_PASSWORD = "admin123"
RTSP_CAMERA_IP = "192.168.128.10"
RTSP_PORT = 554
RTSP_CHANNEL = 1
RTSP_STREAM = 1
SPARSH_RTSP_URL = (
    f"rtsp://{RTSP_USERNAME}:{RTSP_PASSWORD}@{RTSP_CAMERA_IP}:{RTSP_PORT}/"
    f"avstream/channel={RTSP_CHANNEL}/stream={RTSP_STREAM}.sdp"
)

# Severity Configuration for Accident Detection
# BGR Colors for OpenCV
# DETR Class indices: 0: accident, 1: accident, 2: vehicle
SEVERITY_CONFIG: Dict[int, Dict[str, any]] = {
    0: {
        "raw_name": "accident",
        "display_name": "Traffic Accident",
        "short_name": "ACCIDENT",
        "color": (0, 40, 240),      # Bright Red
        "is_accident": True,
        "level": 3
    },
    1: {
        "raw_name": "accident",
        "display_name": "Traffic Accident",
        "short_name": "ACCIDENT",
        "color": (0, 40, 240),      # Bright Red
        "is_accident": True,
        "level": 3
    },
    2: {
        "raw_name": "vehicle",
        "display_name": "Normal Vehicle",
        "short_name": "NORMAL",
        "color": (0, 210, 60),      # Bright Green
        "is_accident": False,
        "level": 0
    },
    3: {
        "raw_name": "accident_severe",
        "display_name": "Severe Accident",
        "short_name": "SEVERE",
        "color": (0, 40, 240),      # Bright Red
        "is_accident": True,
        "level": 3
    },
    4: {
        "raw_name": "accident_100",
        "display_name": "Totaled Vehicle",
        "short_name": "TOTALED",
        "color": (160, 0, 210),     # Magenta
        "is_accident": True,
        "level": 4
    }
}

# Fire & Hazard Detection Classes and Colors
FIRE_CLASSES = ["fire", "other", "smoke"]
FIRE_COLORS = {
    "fire": (0, 69, 255),       # Vibrant Red-Orange
    "smoke": (180, 180, 180),   # Light Gray
    "other": (150, 150, 150)    # Gray
}
FIRE_ALERT_COLOR = (0, 0, 255)       # Red
NORMAL_STATUS_COLOR = (0, 200, 0)   # Green

# Thresholds
DEFAULT_CRASH_CONF = 0.70
DEFAULT_CRASH_IOU = 0.65
DEFAULT_FIRE_CONF = 0.75

# Snapshot cooldown (seconds) to prevent flooding disk during continuous alerts
ALERT_SNAPSHOT_COOLDOWN_SEC = 5.0
