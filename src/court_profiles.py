"""Per-camera-angle calibration, kept separate from the detection algorithm
in court_lines.py so a new camera setup (e.g. a HORIZONTAL view down the
width of the court instead of this one's VERTICAL view down its length)
just needs a new profile here -- no changes to the detection logic itself.
"""

from dataclasses import dataclass

import numpy as np

# Regulation pickleball court, in meters (used to build the top-down reconstruction).
COURT_WIDTH_M = 6.10
HALF_COURT_DEPTH_M = 6.70  # net to baseline, one side
KITCHEN_DEPTH_M = 2.13


@dataclass
class CourtProfile:
    name: str
    orientation: str  # "vertical" (camera looks down the court's length) | "horizontal" (looks across its width)
    line_search_polygon: np.ndarray  # where to look for painted lines
    net_exclude_box: np.ndarray      # net mesh is too noisy for line detection; excluded and drawn separately
    net_left_post: tuple
    net_right_post: tuple
    net_post_height: int
    # HSV range isolating the court's own playing-surface color (e.g. navy blue),
    # used to find the sideline edges directly and reliably near the baseline --
    # much less error-prone than extrapolating a long fitted line.
    court_surface_hsv: tuple  # (lower, upper)
    baseline_row: int         # a y (or x, for a horizontal-orientation profile) near the
                               # baseline, clear of logos/watermarks, to measure the court
                               # surface's edge at


# Calibrated for picklecam.mp4's fixed camera: looking down the court's
# length, net roughly across the middle of frame, one half-court visible
# in front of it. Recalibrate these four values for a different camera.
PICKLECAM_VERTICAL = CourtProfile(
    name="picklecam_vertical",
    orientation="vertical",
    line_search_polygon=np.array([(380, 140), (1480, 140), (1920, 1080), (0, 1080)], dtype=np.int32),
    net_exclude_box=np.array([(400, 150), (1500, 150), (1500, 340), (400, 340)], dtype=np.int32),
    net_left_post=(463, 295),
    net_right_post=(1430, 295),
    net_post_height=65,
    court_surface_hsv=((100, 10, 120), (130, 50, 210)),
    baseline_row=900,
)

PROFILES = {"picklecam_vertical": PICKLECAM_VERTICAL}


def get_profile(name: str) -> CourtProfile:
    if name not in PROFILES:
        raise ValueError(f"Unknown court profile '{name}'. Available: {list(PROFILES)}")
    return PROFILES[name]
