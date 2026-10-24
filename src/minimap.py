"""Small overview panel (top-right corner of the live preview window)
showing the calibrated court zones (e.g. near/far kitchen) in miniature.
Every zone always shows a colored shade on both the minimap AND the main
video (so which-part-is-which is visible even without touching the mouse);
hovering a zone in the minimap brightens that shade further, on both.

Only meaningful in an interactive window (--show): a batch-rendered video
file has no mouse to hover.
"""

import cv2
import numpy as np

MINIMAP_SIZE = (300, 200)  # w, h -- generous target size, easier to hit with the mouse
MINIMAP_MARGIN = 12

# Distinct color per zone (BGR), cycling if there are more zones than colors.
ZONE_COLORS = [
    (255, 140, 0),    # cyan-ish blue
    (180, 0, 200),    # magenta/purple
    (0, 200, 120),    # green
    (0, 140, 255),    # orange
    (200, 200, 0),    # teal
    (60, 60, 220),    # red-ish
    (200, 0, 120),    # violet
    (0, 220, 220),    # yellow
]

IDLE_ALPHA = 0.22
HOVER_ALPHA = 0.55


def _zones_bbox(zones: dict, pad: int = 40):
    xs = [p[0] for pts in zones.values() for p in pts]
    ys = [p[1] for pts in zones.values() for p in pts]
    if not xs:
        return None
    return (min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad)


def _zone_color(index: int):
    return ZONE_COLORS[index % len(ZONE_COLORS)]


class Minimap:
    def __init__(self, zones: dict, frame_shape):
        self.zones = zones
        self.names = list(zones)
        h, w = frame_shape[:2]
        x1, y1 = w - MINIMAP_MARGIN, MINIMAP_MARGIN + MINIMAP_SIZE[1]
        x0, y0 = x1 - MINIMAP_SIZE[0], MINIMAP_MARGIN
        self.rect = (x0, y0, x1, y1)
        self.bbox = _zones_bbox(zones)
        if self.bbox:
            bx0, by0, bx1, by1 = self.bbox
            self.scale_x = MINIMAP_SIZE[0] / max(1, (bx1 - bx0))
            self.scale_y = MINIMAP_SIZE[1] / max(1, (by1 - by0))

    def _to_minimap(self, pt):
        x0, y0, _, _ = self.rect
        bx0, by0, _, _ = self.bbox
        return (x0 + (pt[0] - bx0) * self.scale_x, y0 + (pt[1] - by0) * self.scale_y)

    def hovered_zone(self, mouse_pos):
        if mouse_pos is None or self.bbox is None:
            return None
        mx, my = mouse_pos
        x0, y0, x1, y1 = self.rect
        if not (x0 <= mx <= x1 and y0 <= my <= y1):
            return None
        for name, pts in self.zones.items():
            mm_pts = np.array([self._to_minimap(p) for p in pts], dtype=np.float32)
            if cv2.pointPolygonTest(mm_pts, (float(mx), float(my)), False) >= 0:
                return name
        return None

    def draw(self, frame, hovered: str | None):
        if self.bbox is None:
            return frame
        x0, y0, x1, y1 = (int(v) for v in self.rect)
        panel = frame[y0:y1, x0:x1]
        frame[y0:y1, x0:x1] = (panel * 0.35).astype(panel.dtype)
        cv2.rectangle(frame, (x0, y0), (x1, y1), (210, 210, 210), 1)

        for i, (name, pts) in enumerate(self.zones.items()):
            mm_pts = np.array([self._to_minimap(p) for p in pts], dtype=np.int32)
            color = _zone_color(i)
            is_hover = name == hovered
            overlay = frame.copy()
            cv2.fillPoly(overlay, [mm_pts], color)
            alpha = HOVER_ALPHA if is_hover else IDLE_ALPHA
            frame[:] = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
            cv2.polylines(frame, [mm_pts], True, color, 2 if is_hover else 1, cv2.LINE_AA)
            label_pt = mm_pts.mean(axis=0).astype(int)
            cv2.putText(frame, name.replace("_", " "), tuple(label_pt), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                        (255, 255, 255), 1, cv2.LINE_AA)

        cv2.putText(frame, "court map (hover a zone)", (x0 + 4, y0 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (210, 210, 210), 1, cv2.LINE_AA)
        return frame

    def shade_zones(self, frame):
        """Shades EVERY zone on the real video with its own color, always --
        not mouse-dependent, so which part is which is visible on its own.
        Safe (and intended) to also bake into the written output file, unlike
        the hover boost and minimap panel below."""
        if not self.zones:
            return frame
        overlay = frame.copy()
        for i, pts in enumerate(self.zones.values()):
            arr = np.array(pts, dtype=np.int32)
            cv2.fillPoly(overlay, [arr], _zone_color(i))
        return cv2.addWeighted(overlay, IDLE_ALPHA, frame, 1 - IDLE_ALPHA, 0)

    def hover_boost(self, frame, hovered: str | None):
        """Extra brightening on top of shade_zones() for whichever zone the
        mouse is over -- mouse-dependent, preview-only."""
        if hovered not in self.zones:
            return frame
        overlay = frame.copy()
        arr = np.array(self.zones[hovered], dtype=np.int32)
        idx = self.names.index(hovered)
        cv2.fillPoly(overlay, [arr], _zone_color(idx))
        extra = HOVER_ALPHA - IDLE_ALPHA
        return cv2.addWeighted(overlay, extra, frame, 1 - extra, 0)

    def draw_mouse_debug(self, frame, mouse_pos):
        """Small crosshair at wherever cv2 THINKS the mouse is, so a coordinate
        mismatch (e.g. from Windows display scaling) is visible at a glance --
        if this doesn't track your real cursor, that's the bug."""
        if mouse_pos is None:
            return frame
        x, y = int(mouse_pos[0]), int(mouse_pos[1])
        cv2.drawMarker(frame, (x, y), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
        cv2.putText(frame, f"mouse: {x},{y}", (x + 12, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 1, cv2.LINE_AA)
        return frame
