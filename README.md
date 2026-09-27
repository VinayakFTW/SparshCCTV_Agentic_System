# Sparsh CCTV Real-Time Accident Severity & Fire Detection System

An AI surveillance system for real-time traffic monitoring, vehicle accident severity classification, and fire/hazard detection using deep learning on CCTV video streams and recorded video files.

---

## Models Integrated

1. **Traffic Accident & Vehicle Detector** (`models/crash_classifier/model.safetensors`):
   - Architecture: **Hugging Face DETR (DEtection TRansformer with ResNet-50 backbone)**
   - Format: SafeTensors with local offline `config.json` & `preprocessor_config.json`
   - Real-time CUDA FP16 inference for CCTV camera feeds and recordings
   - Classes detected:
     - `accident` (Traffic Accident - Red badge, Level 3 alert)
     - `vehicle` (Normal Vehicle - Green badge, Level 0 normal)

2. **Fire & Smoke Hazard Detector** (`models/fire_classifier/fire_classifier.pt`):
   - Architecture: **Ultralytics YOLO** Object Detector
   - Classes: `0: fire`, `1: other`, `2: smoke`
   - Real-time bounding box detection of flames and smoke plumes with high-priority emergency alerts and border flashes.

---

## Features

- **Multi-Source Support**:
  - Video files (`.mp4`, `.avi`, `.mkv`, etc.)
  - Sparsh CCTV RTSP live stream (`rtsp://admin:admin123@192.168.128.10:554/...`)
  - Webcams (source index `0`, `1`, etc.)
- **Surveillance Camera HUD**:
  - Live FPS counter & timestamp.
  - Active vehicle counter & categorized incident breakdown.
  - Color-coded bounding boxes with confidence scores.
  - Automatic flashing alert banners for severe accidents and fires.
- **Incident Logging & Snapshots**:
  - Automatically captures high-resolution evidence snapshots (`output/snapshots/`) when severe accidents or fires occur (with cooldown to prevent spam).
  - Logs structured event data to `output/incident_log.csv`.
- **Interactive Controls**:
  - `Space`: Pause / Resume playback
  - `S`: Save manual snapshot to `output/snapshots/`
  - `H`: Toggle HUD overlay on/off
  - `Q` or `ESC`: Quit cleanly
- **Video Recording**:
  - Optional `--save` flag to export processed videos with full HUD overlay.

---

## Quick Start & Usage

### 1. Run on a Video File
```powershell
.venv/Scripts/python.exe main.py --source data/videos/v3.mp4
```

### 2. Save Processed Video to Disk
```powershell
.venv/Scripts/python.exe main.py --source data/videos/v3.mp4 --save output/recordings/v3_annotated.mp4
```

### 3. Connect to Sparsh CCTV RTSP Stream
```powershell
.venv/Scripts/python.exe main.py --rtsp
```

### 4. Adjust Detection Thresholds
```powershell
.venv/Scripts/python.exe main.py --source data/videos/v3.mp4 --conf 0.40 --fire-conf 0.70
```

### 5. Run Headless (No GUI Window)
```powershell
.venv/Scripts/python.exe --source data/videos/v3.mp4 --no-display --save
```

---

## Automated Tests

Run the test suite to verify model loading, synthetic inference, visualizer rendering, and video capture:
```powershell
.venv/Scripts/python.exe tests/test_pipeline.py
```

---

## Project Structure

```
SparshCCTV_Agentic_System/
├── config.py                  # System configurations, RTSP credentials, thresholds, colors
├── main.py                    # Main CLI application entry point
├── stream_capture.py          # Video stream capture handler (RTSP & video files)
├── visualizer.py              # CCTV HUD overlay and incident rendering engine
├── incident_logger.py         # Snapshot capturer and CSV incident logging
├── detectors/
│   ├── accident_detector.py   # DETR ResNet-50 accident & vehicle detector
│   ├── fire_detector.py       # Ultralytics YOLO fire & smoke detector
│   └── __init__.py            # Detector module package exports
├── models/
│   ├── crash_classifier/      # DETR crash model weights (model.safetensors & configs)
│   └── fire_classifier/       # YOLO fire & smoke model weights (fire_classifier.pt)
├── output/
│   ├── snapshots/             # Saved incident images
│   └── recordings/            # Saved video recordings
└── tests/
    └── test_pipeline.py       # Automated unit and integration tests
```
