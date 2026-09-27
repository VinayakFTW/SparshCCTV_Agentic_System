import argparse
import os
import sys
import time
import cv2
import torch

from config import (
    DEFAULT_TEST_VIDEO,
    SPARSH_RTSP_URL,
    DEFAULT_CRASH_CONF,
    DEFAULT_FIRE_CONF,
    RECORDING_DIR,
    ACCIDENT_DETECTOR_BACKEND,
    DEFAULT_CAMERA_FPS
)
from stream_capture import StreamCapture
from detectors.accident_detector import AccidentDetector
from detectors.yolo_hierarchical_detector import YOLOHierarchicalAccidentDetector
from detectors.fire_detector import FireDetector
from visualizer import CCTVVisualizer
from incident_logger import IncidentLogger


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Sparsh CCTV Real-Time Accident Severity & Fire Detection System"
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Path to video file, webcam index (e.g. 0), or RTSP URL. Defaults to data/videos/v3.mp4 if not specified."
    )
    parser.add_argument(
        "--rtsp",
        action="store_true",
        help="Stream directly from Sparsh CCTV RTSP camera feed."
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=DEFAULT_CRASH_CONF,
        help=f"Confidence threshold for accident/vehicle detection (default: {DEFAULT_CRASH_CONF})"
    )
    parser.add_argument(
        "--fire-conf",
        type=float,
        default=DEFAULT_FIRE_CONF,
        help=f"Confidence threshold for fire detection (default: {DEFAULT_FIRE_CONF})"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on: 'cuda' or 'cpu'"
    )
    parser.add_argument(
        "--save",
        type=str,
        nargs="?",
        const=os.path.join(RECORDING_DIR, "surveillance_output.mp4"),
        default=None,
        help="Path to save output video with HUD. If passed without value, saves to output/recordings/surveillance_output.mp4"
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Run headless without opening OpenCV preview window."
    )
    parser.add_argument(
        "--detector-backend",
        type=str,
        default=ACCIDENT_DETECTOR_BACKEND,
        choices=["yolo", "detr"],
        help=f"Accident detector backend: 'yolo' (Hierarchical 15 FPS) or 'detr' (legacy) (default: {ACCIDENT_DETECTOR_BACKEND})"
    )
    parser.add_argument(
        "--fps-cap",
        type=float,
        default=DEFAULT_CAMERA_FPS,
        help=f"Target camera stream FPS cap for real-time synchronization (default: {DEFAULT_CAMERA_FPS})"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional maximum number of frames to process."
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    # Determine input source
    if args.rtsp:
        video_source = SPARSH_RTSP_URL
        source_label = "Sparsh CCTV (Live RTSP)"
    elif args.source:
        # Check if integer webcam index
        if args.source.isdigit():
            video_source = int(args.source)
            source_label = f"Webcam {video_source}"
        else:
            video_source = args.source
            source_label = os.path.basename(args.source)
    else:
        video_source = DEFAULT_TEST_VIDEO
        source_label = os.path.basename(DEFAULT_TEST_VIDEO)

    print("=" * 65)
    print("   SPARSH CCTV REAL-TIME ACCIDENT SEVERITY & FIRE DETECTOR   ")
    print("=" * 65)
    print(f"[*] Input Source    : {video_source} ({source_label})")
    print(f"[*] Detector Backend: {args.detector_backend.upper()}")
    print(f"[*] Target Stream   : {args.fps_cap} FPS Cap")
    print(f"[*] Inference Device: {args.device}")
    print(f"[*] Crash Conf Thres: {args.conf}")
    print(f"[*] Fire Conf Thres : {args.fire_conf}")
    print(f"[*] Save Video      : {args.save if args.save else 'Disabled'}")
    print(f"[*] GUI Preview     : {'Disabled (Headless)' if args.no_display else 'Enabled'}")
    print("=" * 65)

    # 1. Initialize Detectors
    if args.detector_backend == "yolo":
        print("[1/4] Initializing Hierarchical Accident & Severity Detector (YOLO11 CUDA FP16)...")
        accident_detector = YOLOHierarchicalAccidentDetector(conf_thres=args.conf, device=args.device)
    else:
        print("[1/4] Initializing Accident Severity Detector (DETR SafeTensors FP16)...")
        accident_detector = AccidentDetector(conf_thres=args.conf, device=args.device)

    print("[2/4] Initializing Fire & Smoke Detector (YOLO)...")
    fire_detector = FireDetector(conf_thres=args.fire_conf, device=args.device)

    # 2. Initialize Visualizer & Logger
    print("[3/4] Initializing Surveillance HUD & Incident Logger...")
    visualizer = CCTVVisualizer(camera_name="SPARSH CCTV 01")
    incident_logger = IncidentLogger()

    # 3. Open Video Stream
    print(f"[4/4] Opening Video Stream from {video_source} (capped at {args.fps_cap} FPS)...")
    try:
        stream = StreamCapture(video_source, name=source_label, target_fps=args.fps_cap)
    except Exception as e:
        print(f"[ERROR] Failed to start stream: {e}")
        sys.exit(1)

    print(f"[+] Video Stream Connected: {stream.width}x{stream.height} @ {stream.fps:.1f} FPS")
    if stream.total_frames > 0:
        print(f"[+] Total Video Frames : {stream.total_frames}")

    # Optional Video Writer
    video_writer = None
    if args.save:
        os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(
            args.save,
            fourcc,
            stream.fps,
            (stream.width, stream.height)
        )
        print(f"[+] Recording output enabled: {args.save}")

    window_name = "Sparsh CCTV - Real-Time Accident & Fire Surveillance"
    if not args.no_display:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, min(1280, stream.width), min(720, stream.height))

    print("\n[+] System running. Press 'q' to quit, 'space' to pause, 's' for snapshot, 'h' for HUD toggle.\n")

    start_time = time.time()
    processed_count = 0
    paused = False

    try:
        while True:
            if not paused:
                ret, frame = stream.read()
                if not ret:
                    print("\n[Info] End of video stream or connection closed.")
                    break

                processed_count += 1
                if args.max_frames and processed_count > args.max_frames:
                    print(f"\n[Info] Reached max frames limit ({args.max_frames}).")
                    break

                # Step 1: Accident and vehicle detection
                detections = accident_detector.detect(frame)
                highest_sev, highest_det = accident_detector.get_highest_severity(detections)

                # Step 2: Fire classification
                fire_result = fire_detector.detect(frame)

                # Step 3: Check and trigger incident snapshot
                incident_logger.check_and_log(
                    frame=frame,
                    highest_severity=highest_sev,
                    highest_det=highest_det,
                    fire_result=fire_result,
                    detections=detections
                )

                # Step 4: Render surveillance HUD overlay
                annotated_frame = visualizer.draw(
                    frame=frame,
                    detections=detections,
                    fire_result=fire_result,
                    source_label=source_label
                )

                # Step 5: Save frame if recording
                if video_writer is not None:
                    video_writer.write(annotated_frame)

            # Step 6: Display frame
            if not args.no_display:
                cv2.imshow(window_name, annotated_frame)
                key = cv2.waitKey(1 if not paused else 30) & 0xFF

                if key == ord('q') or key == 27:  # 'q' or ESC
                    print("\n[User] Exit command received.")
                    break
                elif key == ord(' '):  # Spacebar
                    paused = not paused
                    print(f"[User] Playback {'PAUSED' if paused else 'RESUMED'}")
                elif key == ord('s') or key == ord('S'):  # Save manual snapshot
                    snap_path = incident_logger.save_snapshot(annotated_frame, tag="manual")
                    print(f"[Snapshot] Manual snapshot saved to: {snap_path}")
                elif key == ord('h') or key == ord('H'):  # Toggle HUD
                    visualizer.show_hud = not visualizer.show_hud
                    print(f"[HUD] Overlay {'ENABLED' if visualizer.show_hud else 'DISABLED'}")

    except KeyboardInterrupt:
        print("\n[Interrupted] Surveillance terminated by user.")

    finally:
        total_time = time.time() - start_time
        stream.release()
        if video_writer is not None:
            video_writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()

        avg_fps = (processed_count / total_time) if total_time > 0 else 0.0
        print("\n" + "=" * 65)
        print("                 EXECUTION SUMMARY")
        print("=" * 65)
        print(f"Total Frames Processed : {processed_count}")
        print(f"Total Execution Time   : {total_time:.2f} seconds")
        print(f"Average Pipeline Speed : {avg_fps:.2f} FPS")
        if args.save:
            print(f"Saved Output Video     : {args.save}")
        print("=" * 65)


if __name__ == "__main__":
    main()