"""Detects the pickleball (COCO 'sports ball' class) each frame and draws a
fading, dashed trail of its recent flight path. YOLO's default inference
resolution (640) is too coarse for an object this small -- it needs a
dedicated higher-resolution pass (imgsz=1280 by default here) separate from
the person/pose model.
"""

import math
from collections import deque

import cv2

SPORTS_BALL_CLASS = 32


class BallTracker:
    def __init__(self, fps: float = 30.0, trail_seconds: float = 2.0, conf: float = 0.3, imgsz: int = 1280,
                 max_gap_seconds: float = 0.2):
        self.conf = conf
        self.imgsz = imgsz
        self.trail_frames = max(1, round(fps * trail_seconds))
        self.max_gap_frames = max(1, round(fps * max_gap_seconds))
        self.positions = deque()  # (frame_idx, x, y)

    def detect(self, model, frame, frame_idx: int):
        """Runs the ball-only detection pass and records the single best hit."""
        results = model.predict(frame, classes=[SPORTS_BALL_CLASS], conf=self.conf, imgsz=self.imgsz, verbose=False)[0]
        if len(results.boxes):
            best = max(results.boxes, key=lambda b: b.conf.item())
            x1, y1, x2, y2 = best.xyxy[0].tolist()
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            self.positions.append((frame_idx, cx, cy))

        while self.positions and frame_idx - self.positions[0][0] > self.trail_frames:
            self.positions.popleft()

    @staticmethod
    def _dashed_line(img, pt1, pt2, color, thickness=2, dash_len=10, gap_len=8):
        x1, y1 = pt1
        x2, y2 = pt2
        dist = math.hypot(x2 - x1, y2 - y1)
        if dist < 1:
            return
        step = dash_len + gap_len
        n = int(dist / step) + 1
        for i in range(n):
            s = min(1.0, (i * step) / dist)
            e = min(1.0, (i * step + dash_len) / dist)
            if s >= e:
                continue
            sx, sy = int(x1 + (x2 - x1) * s), int(y1 + (y2 - y1) * s)
            ex, ey = int(x1 + (x2 - x1) * e), int(y1 + (y2 - y1) * e)
            cv2.line(img, (sx, sy), (ex, ey), color, thickness, cv2.LINE_AA)

    def draw(self, frame, frame_idx: int, color=(0, 215, 255)):
        pts = list(self.positions)
        if len(pts) >= 2:
            for i in range(1, len(pts)):
                f0, x0, y0 = pts[i - 1]
                f1, x1, y1 = pts[i]
                if f1 - f0 > self.max_gap_frames:
                    # Ball wasn't detected for a while between these two points --
                    # they're not really the same continuous flight, so don't draw
                    # a line bridging them. Just skip to the next pair.
                    continue
                age = frame_idx - f1
                fade = max(0.0, 1.0 - age / self.trail_frames)
                if fade <= 0.05:
                    continue
                overlay = frame.copy()
                self._dashed_line(overlay, (x0, y0), (x1, y1), color, thickness=2)
                frame = cv2.addWeighted(overlay, 0.6 * fade, frame, 1 - 0.6 * fade, 0)

        if pts:
            _, cx, cy = pts[-1]
            if frame_idx - pts[-1][0] <= 2:
                cv2.circle(frame, (int(cx), int(cy)), 5, color, -1, cv2.LINE_AA)

        return frame
