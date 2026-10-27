"""Detects the pickleball (COCO 'sports ball' class) each frame and draws a
fading, dashed trail of its recent flight path. YOLO's default inference
resolution (640) is too coarse for an object this small -- it needs a
dedicated higher-resolution pass (imgsz=1280 by default here) separate from
the person/pose model.

When YOLO misses a frame (common for a small, fast, motion-blurred ball),
a background-subtraction + HSV-color fallback (same recipe as the
PickleJudge project's detector-cpu.ts: color is never a gate on its own --
a static ball-colored line/shirt has no motion, a moving non-ball object
has the wrong color) tries to recover the position instead of leaving a
gap in the trail.
"""

import math
from collections import deque

import cv2
import numpy as np

SPORTS_BALL_CLASS = 32


class BallTracker:
    def __init__(self, fps: float = 30.0, trail_seconds: float = 2.0, conf: float = 0.3, imgsz: int = 1280,
                 max_gap_seconds: float = 0.2, small_width: int = 480, bg_alpha: float = 0.03,
                 motion_thresh: float = 18.0, hue_range: tuple = (18, 48), sat_min: int = 70, val_min: int = 90,
                 min_blob_area: int = 2, max_blob_area: int = 250, max_aspect: float = 2.5,
                 max_speed_px_per_sec: float = 5000.0):
        self.conf = conf
        self.imgsz = imgsz
        self.fps = fps
        self.trail_frames = max(1, round(fps * trail_seconds))
        self.max_gap_frames = max(1, round(fps * max_gap_seconds))
        # Caps how far a new hit may sit from the last known position, scaled
        # by how many frames have passed -- rejects teleporting picks (a
        # color+motion fallback catching a different yellow object, or a bad
        # YOLO box) that would otherwise get wired into the trail as a single
        # continuous "flight" no real ball could make.
        self.max_px_per_frame = max_speed_px_per_sec / fps
        self.positions = deque()  # (frame_idx, x, y)

        # Color+motion fallback state (see _color_motion_mask / _select_blob).
        self.small_width = small_width
        self.bg_alpha = bg_alpha
        self.motion_thresh = motion_thresh
        self.hue_range = hue_range
        self.sat_min = sat_min
        self.val_min = val_min
        self.min_blob_area = min_blob_area
        self.max_blob_area = max_blob_area
        self.max_aspect = max_aspect
        self._bg = None
        self._last_pos = None  # (frame_idx, x, y) -- most recent hit, either source

    def detect(self, model, frame, frame_idx: int):
        """Runs the ball-only YOLO pass; if it finds nothing, falls back to a
        color+motion blob pick before giving up on the frame entirely."""
        results = model.predict(frame, classes=[SPORTS_BALL_CLASS], conf=self.conf, imgsz=self.imgsz, verbose=False)[0]
        hit = None
        if len(results.boxes):
            best = max(results.boxes, key=lambda b: b.conf.item())
            x1, y1, x2, y2 = best.xyxy[0].tolist()
            hit = ((x1 + x2) / 2, (y1 + y2) / 2)
        else:
            hit = self._detect_color_motion(frame, frame_idx)

        if hit is not None and not self._is_implausible_jump(frame_idx, hit):
            self.positions.append((frame_idx, hit[0], hit[1]))
            self._last_pos = (frame_idx, hit[0], hit[1])

        while self.positions and frame_idx - self.positions[0][0] > self.trail_frames:
            self.positions.popleft()

    def _is_implausible_jump(self, frame_idx: int, hit) -> bool:
        """True if `hit` is farther from the last known position than a real
        ball could travel in the elapsed frames -- a sign the detector (YOLO
        or the color+motion fallback) latched onto a different object."""
        if self._last_pos is None:
            return False
        last_frame, lx, ly = self._last_pos
        gap = frame_idx - last_frame
        if gap <= 0:
            return False
        allowed = self.max_px_per_frame * gap
        return math.hypot(hit[0] - lx, hit[1] - ly) > allowed

    def _color_motion_mask(self, frame):
        """Downscaled motion-AND-color mask. Returns (mask, scale) or
        (None, scale) during the one-frame background warm-up."""
        h, w = frame.shape[:2]
        scale = max(1, round(w / self.small_width))
        sw, sh = max(1, w // scale), max(1, h // scale)
        small = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)

        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hue, sat, val = hsv[..., 0].astype(np.int32), hsv[..., 1], hsv[..., 2]
        lo, hi = self.hue_range
        color_mask = (hue >= lo) & (hue <= hi) if lo <= hi else (hue >= lo) | (hue <= hi)
        color_mask &= (sat >= self.sat_min) & (val >= self.val_min)

        if self._bg is None or self._bg.shape != gray.shape:
            self._bg = gray.copy()
            return None, scale

        motion_mask = np.abs(gray - self._bg) > self.motion_thresh
        self._bg += self.bg_alpha * (gray - self._bg)

        return (motion_mask & color_mask).astype(np.uint8), scale

    def _select_blob(self, mask, scale, predicted):
        """Picks the best blob by shape/size, preferring one close to
        `predicted` (the last known ball position) when there's more than
        one color+motion match on the frame."""
        num, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        best, best_score = None, -1.0
        for i in range(1, num):  # label 0 is background
            area = stats[i, cv2.CC_STAT_AREA]
            if not (self.min_blob_area <= area <= self.max_blob_area):
                continue
            box_w, box_h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            if max(box_w, box_h) / max(1, min(box_w, box_h)) > self.max_aspect:
                continue  # a ball is roughly round; reject thin line fragments
            cx, cy = centroids[i][0] * scale, centroids[i][1] * scale
            score = float(area)
            if predicted is not None:
                score -= 0.5 * math.hypot(cx - predicted[0], cy - predicted[1])
            if score > best_score:
                best_score, best = score, (cx, cy)
        return best

    def _detect_color_motion(self, frame, frame_idx: int):
        mask, scale = self._color_motion_mask(frame)
        if mask is None:
            return None
        predicted = None
        if self._last_pos is not None and frame_idx - self._last_pos[0] <= self.max_gap_frames:
            predicted = (self._last_pos[1], self._last_pos[2])
        return self._select_blob(mask, scale, predicted)

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
                if f1 - f0 != 1:
                    # Not two back-to-back detections -- the ball dropped out
                    # for one or more frames in between, so this isn't really
                    # one continuous flight. Connecting them anyway draws a
                    # straight "teleport" line across whatever happened during
                    # the gap, so skip it: only draw segments between frames
                    # the ball was actually seen on, one after another.
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
