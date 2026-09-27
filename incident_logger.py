import csv
import os
import time
from datetime import datetime
from typing import List, Optional
import cv2
import numpy as np

from config import SNAPSHOT_DIR, OUTPUT_DIR, ALERT_SNAPSHOT_COOLDOWN_SEC
from detectors.accident_detector import VehicleDetection
from detectors.fire_detector import FireResult


class IncidentLogger:
    """Manages snapshot logging and event recording for surveillance incidents."""

    def __init__(self, log_file: Optional[str] = None):
        self.log_file = log_file or os.path.join(OUTPUT_DIR, "incident_log.csv")
        self.last_accident_alert_time = 0.0
        self.last_fire_alert_time = 0.0
        self.cooldown = ALERT_SNAPSHOT_COOLDOWN_SEC

        self._init_csv()

    def _init_csv(self):
        """Initializes the CSV incident log file if not already present."""
        if not os.path.exists(self.log_file):
            with open(self.log_file, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Timestamp",
                    "Event_Type",
                    "Max_Accident_Severity",
                    "Fire_Probability",
                    "Vehicles_Count",
                    "Snapshot_Path"
                ])

    def save_snapshot(self, frame: np.ndarray, tag: str = "manual") -> str:
        """Saves a timestamped JPEG image snapshot to disk."""
        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
        filename = f"{tag}_{ts_str}.jpg"
        filepath = os.path.join(SNAPSHOT_DIR, filename)
        cv2.imwrite(filepath, frame)
        return filepath

    def check_and_log(
        self,
        frame: np.ndarray,
        highest_severity: int,
        highest_det: Optional[VehicleDetection],
        fire_result: Optional[FireResult],
        detections: List[VehicleDetection]
    ) -> Optional[str]:
        """
        Evaluates current frame state. If an incident or fire is detected and
        cooldown period has elapsed, saves snapshot and logs event.
        """
        now = time.time()
        triggered_events = []

        # Check Fire & Smoke Alert
        if fire_result:
            if fire_result.has_fire:
                if now - self.last_fire_alert_time > self.cooldown:
                    self.last_fire_alert_time = now
                    triggered_events.append("FIRE_ALERT")
            elif fire_result.has_smoke:
                if now - self.last_fire_alert_time > self.cooldown:
                    self.last_fire_alert_time = now
                    triggered_events.append("SMOKE_ALERT")

        # Check Severe Accident (Severity >= 2: Moderate, Severe, Totaled)
        if highest_severity >= 2:
            if now - self.last_accident_alert_time > self.cooldown:
                self.last_accident_alert_time = now
                triggered_events.append(f"ACCIDENT_LVL_{highest_severity}")

        if not triggered_events:
            return None

        event_str = "_".join(triggered_events)
        snapshot_path = self.save_snapshot(frame, tag=event_str.lower())

        # Write to CSV log
        try:
            with open(self.log_file, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    event_str,
                    highest_det.raw_name if highest_det else "None",
                    f"{fire_result.fire_probability:.3f}" if fire_result else "0.0",
                    len(detections),
                    snapshot_path
                ])
            print(f"[ALERT LOGGED] Event: {event_str} -> Snapshot: {snapshot_path}")
        except Exception as e:
            print(f"[IncidentLogger] Error saving log: {e}")

        return snapshot_path
