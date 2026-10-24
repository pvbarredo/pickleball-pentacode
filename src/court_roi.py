"""Restricts detections to the featured/main court, excluding background
courts, spectators, and staff walking along the walls. Tuned for the fixed
PickleCam angle in picklecam.mp4 (1920x1080) -- adjust the polygon if the
camera or venue changes.
"""

import cv2
import numpy as np

MAIN_COURT_POLYGON = np.array(
    [
        (460, 270),   # near net post, left
        (1430, 270),  # near net post, right
        (1920, 1080), # bottom-right of frame
        (0, 1080),    # bottom-left of frame
    ],
    dtype=np.int32,
)


def in_court(box_xyxy, polygon: np.ndarray = MAIN_COURT_POLYGON) -> bool:
    x1, y1, x2, y2 = box_xyxy
    foot_point = ((x1 + x2) / 2.0, y2)
    return cv2.pointPolygonTest(polygon, foot_point, False) >= 0
