"""Flags non-volley-zone (kitchen) rule violations -- a player's foot in the
kitchen, and specifically a volley (swinging while a foot is in) -- by
reusing pieces this project already has: the court-zone polygons from
calibration (same point-in-polygon test the minimap already uses) and the
--pose skeleton's ankle/wrist keypoints. No new detection model.

Swing detection is a simple heuristic (burst of wrist-keypoint movement
over a short window), loosely adapted from the pose-based foot-fault/
kitchen-violation logic in
https://github.com/arcotn/smart-pickleball-assistant (pose_analyzer.py),
rewired here onto this project's own zones and BoT-SORT track ids instead
of a separate court-zone UI and DeepSORT.
"""

from collections import defaultdict, deque

import cv2
import numpy as np

LEFT_ANKLE, RIGHT_ANKLE = 15, 16
LEFT_WRIST, RIGHT_WRIST = 9, 10


class ViolationDetector:
    def __init__(self, kitchen_zones: dict, swing_window: int = 8, swing_thresh: float = 25.0,
                 min_kpt_conf: float = 0.3, cooldown_frames: int = 15):
        """`kitchen_zones`: the subset of load_calibration()'s zones dict
        whose name contains "kitchen" (near_kitchen / far_kitchen)."""
        self.kitchen_zones = {name: np.array(pts, dtype=np.float32) for name, pts in kitchen_zones.items()}
        self.swing_window = swing_window
        self.swing_thresh = swing_thresh
        self.min_kpt_conf = min_kpt_conf
        self.cooldown_frames = cooldown_frames
        self._wrist_history = defaultdict(lambda: deque(maxlen=swing_window))
        self._last_flagged = {}  # track_id -> frame_idx, avoids re-flagging every frame of one violation

    def _in_any_kitchen(self, pt):
        for name, poly in self.kitchen_zones.items():
            if cv2.pointPolygonTest(poly, (float(pt[0]), float(pt[1])), False) >= 0:
                return name
        return None

    def _is_swinging(self, track_id: int) -> bool:
        pts = list(self._wrist_history[track_id])
        if len(pts) < 3:
            return False
        steps = [np.linalg.norm(pts[i] - pts[i - 1]) for i in range(1, len(pts))]
        return max(steps) > self.swing_thresh

    def update(self, track_id: int, kpts_xy: np.ndarray, kpts_conf, frame_idx: int):
        """Call once per tracked person per frame (needs --pose keypoints).
        Returns a violation dict, or None if this track has no violation
        right now (or one was already flagged recently -- cooldown)."""
        if not self.kitchen_zones or kpts_xy is None:
            return None

        def ok(i):
            return kpts_conf is None or kpts_conf[i] >= self.min_kpt_conf

        wrist_pts = [kpts_xy[i] for i in (LEFT_WRIST, RIGHT_WRIST) if ok(i)]
        if wrist_pts:
            self._wrist_history[track_id].append(np.mean(wrist_pts, axis=0))

        ankle_zone = None
        for idx in (LEFT_ANKLE, RIGHT_ANKLE):
            if ok(idx):
                zone = self._in_any_kitchen(kpts_xy[idx])
                if zone:
                    ankle_zone = zone
                    break
        if ankle_zone is None:
            return None

        last = self._last_flagged.get(track_id)
        if last is not None and frame_idx - last < self.cooldown_frames:
            return None

        swinging = self._is_swinging(track_id)
        self._last_flagged[track_id] = frame_idx
        return {
            "frame": frame_idx,
            "track_id": track_id,
            "zone": ankle_zone,
            "type": "kitchen_volley" if swinging else "foot_fault",
            "severity": "high" if swinging else "low",
        }


def draw_violation(frame, violation: dict, bbox, name: str | None = None):
    """Red box + label over the offending player's current bbox."""
    x1, y1, x2, y2 = (int(v) for v in bbox)
    label = "KITCHEN VOLLEY" if violation["type"] == "kitchen_volley" else "FOOT FAULT"
    color = (0, 0, 255) if violation["severity"] == "high" else (0, 140, 255)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3, cv2.LINE_AA)
    tag = f"{label} ({name})" if name else label
    cv2.putText(frame, tag, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return frame
