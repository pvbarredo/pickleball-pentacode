"""Detects the painted court lines (sidelines, baseline, center line, kitchen
/ non-volley-zone lines) once from a calibration frame, since the camera is
fixed and the lines don't move. Also draws the net as a simple calibrated
line + posts, since the net mesh is too noisy for line detection to isolate
cleanly. The result is cached and redrawn as an overlay on every frame.

Raw Hough output is a pile of short, jagged, overlapping fragments. But every
real court line is the edge of a rectangle (the court boundary, the kitchen,
the service boxes), so fragments belonging to the same real edge are grouped
together, fit into one clean line, and extended out to its natural full
length -- rather than drawn as-is.

Everything here is generic (works on any camera angle); what's specific to
ONE camera setup -- the search region, the net's pixel position -- lives in
court_profiles.py as a swappable CourtProfile.
"""

import math

import cv2
import numpy as np

from src.court_profiles import CourtProfile


def _detect_raw_segments(frame, profile: CourtProfile, min_line_length: int = 130, max_line_gap: int = 20) -> list:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    white_mask = cv2.inRange(hsv, (0, 0, 180), (180, 60, 255))

    roi = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(roi, [profile.line_search_polygon], 255)
    cv2.fillPoly(roi, [profile.net_exclude_box], 0)
    white_mask = cv2.bitwise_and(white_mask, white_mask, mask=roi)

    edges = cv2.Canny(white_mask, 50, 150)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=60,
        minLineLength=min_line_length, maxLineGap=max_line_gap,
    )
    if lines is None:
        return []
    return [tuple(int(v) for v in l.ravel()) for l in lines]


def _cluster_and_merge(segments: list, angle_tol_deg: float = 7.0, offset_tol: float = 35.0) -> list:
    """Groups fragments that belong to the same real rectangle edge (close in
    angle and perpendicular offset), fits one line through each group, and
    extends it to that group's own natural extent -- turning a pile of short
    Hough fragments into a handful of clean, continuous lines."""
    clusters = []  # each: {"angle": deg, "offset": float, "points": [(x,y), ...]}

    for x1, y1, x2, y2 in segments:
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length < 1:
            continue
        angle = math.degrees(math.atan2(dy, dx)) % 180
        nx, ny = -dy / length, dx / length  # unit normal
        offset = nx * x1 + ny * y1  # signed perpendicular distance from origin

        match = None
        for c in clusters:
            angle_diff = min(abs(angle - c["angle"]), 180 - abs(angle - c["angle"]))
            if angle_diff <= angle_tol_deg and abs(offset - c["offset"]) <= offset_tol:
                match = c
                break

        if match is None:
            clusters.append({"angle": angle, "offset": offset, "points": [(x1, y1), (x2, y2)]})
        else:
            match["points"].extend([(x1, y1), (x2, y2)])
            n = len(match["points"])
            match["angle"] += (angle - match["angle"]) / n
            match["offset"] += (offset - match["offset"]) / n

    merged = []
    for c in clusters:
        pts = np.array(c["points"], dtype=np.float32)
        vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()

        # The group's own natural extent, not the frame edges (extending to the
        # frame bounds sent lines shooting off into the ceiling/walls). Just
        # close the small gaps between fragments of the same real edge.
        proj = (pts[:, 0] - x0) * vx + (pts[:, 1] - y0) * vy
        margin = 15
        t_min, t_max = proj.min() - margin, proj.max() + margin

        p1 = (x0 + vx * t_min, y0 + vy * t_min)
        p2 = (x0 + vx * t_max, y0 + vy * t_max)
        merged.append((int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1])))
    return merged


def detect_court_lines(frame, profile: CourtProfile) -> list:
    """Detect the painted white court lines once on a representative frame,
    returning clean full-length lines (not raw jagged fragments)."""
    raw = _detect_raw_segments(frame, profile)
    return _cluster_and_merge(raw)


def detect_court_lines_stable(cap_path: str, profile: CourtProfile, sample_frames: int = 5) -> list:
    """Tries a few spread-out frames and keeps the single CLEANEST result (the
    merge with the fewest lines) -- more robust than hardcoding one frame
    index where a player might happen to be standing on a line.

    Pooling fragments from every sampled frame together was tried first and
    made things worse, not better: each frame's fragments for the same real
    edge land at a *slightly* different angle/offset (motion blur, a
    different player's shadow, JPEG noise), so pooling multiplies the number
    of near-duplicate clusters instead of reducing them. One frame's own
    fragments are self-consistent, so merge each frame separately and just
    pick the least cluttered result."""
    cap = cv2.VideoCapture(cap_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    candidates = []
    for i in range(sample_frames):
        frame_idx = int(total * (i + 1) / (sample_frames + 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if ok:
            candidates.append(_cluster_and_merge(_detect_raw_segments(frame, profile)))
    cap.release()
    if not candidates:
        return []
    return min(candidates, key=len)


def _dist(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _order_from_to(points, start_ref, end_ref):
    """Orders points to run from whichever end is closer to start_ref, toward
    end_ref (sorts by y as a baseline, since these connector segments run
    roughly net-to-kitchen-line, then flips if that put the wrong end first)."""
    pts = sorted(points, key=lambda p: p[1])
    if _dist(pts[0], start_ref) > _dist(pts[-1], start_ref):
        pts = list(reversed(pts))
    return pts


def _closest_point_on_segment(p, a, b):
    ax, ay = a
    bx, by = b
    px, py = p
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return a
    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    return (ax + t * dx, ay + t * dy)


def _net_merge_point(net_a, net_b, side_points):
    """Where a sideline actually merges into the net's base line -- the point
    ON the net's base line nearest to the sideline's own traced curve --
    instead of just using wherever the net's corner happened to be clicked
    (which may not land exactly on the true sideline)."""
    if not side_points:
        return None
    nearest_side_pt = min(side_points, key=lambda sp: _dist(sp, _closest_point_on_segment(sp, net_a, net_b)))
    x, y = _closest_point_on_segment(nearest_side_pt, net_a, net_b)
    return (int(x), int(y))  # cv2.line needs int coordinates; the segment-projection math is float


def _build_kitchen_zone(net_base_left, net_base_right, kitchen_points, left_side_points, right_side_points):
    """Closes a kitchen/non-volley-zone polygon using the net's BASE (where it
    meets the floor -- the shared ground-level boundary between the near and
    far halves, used for BOTH zones), the kitchen line's own traced curve, and
    -- between the two -- the ACTUAL traced sideline, not a straight synthetic
    connector. Rather than starting from the net's literal clicked corner
    (which may not sit exactly on the true sideline), each side starts from
    the computed MERGE point: wherever that sideline's own curve actually
    comes closest to the net's base line.

    (The net's top-cable corners are NOT used here: they're elevated by the
    net's height, which is irrelevant to a floor-level zone, and using them
    for the far zone was an earlier bug that produced a twisted/self-crossing
    shape.)"""
    by_x = sorted(kitchen_points, key=lambda p: p[0])
    kitchen_left_end, kitchen_right_end = by_x[0], by_x[-1]

    merge_left = _net_merge_point(net_base_left, net_base_right, left_side_points) or net_base_left
    merge_right = _net_merge_point(net_base_left, net_base_right, right_side_points) or net_base_right

    y_vals = [merge_left[1], merge_right[1]] + [p[1] for p in kitchen_points]
    y_lo, y_hi = min(y_vals) - 15, max(y_vals) + 15
    y_lo, y_hi = min(y_lo, y_hi), max(y_lo, y_hi)

    polygon = [merge_left]
    if left_side_points:
        between = [p for p in left_side_points if y_lo <= p[1] <= y_hi]
        if between:
            polygon += _order_from_to(between, merge_left, kitchen_left_end)
    polygon += by_x
    if right_side_points:
        between = [p for p in right_side_points if y_lo <= p[1] <= y_hi]
        if between:
            polygon += _order_from_to(between, kitchen_right_end, merge_right)
    polygon.append(merge_right)
    return polygon


def _split_at_x(points, x_split):
    left = sorted([p for p in points if p[0] <= x_split], key=lambda p: p[0])
    right = sorted([p for p in points if p[0] > x_split], key=lambda p: p[0])
    return left, right


def _build_service_zones(kitchen_points, baseline_points, center_points, left_side_points, right_side_points):
    """Closes the two service-box polygons (left/right of the center line, between
    the kitchen line and the baseline) out of already-traced, trusted lines --
    no new geometry is invented, only connected. Skips a side if its sideline
    wasn't traced (nothing to close it with)."""
    if not (kitchen_points and baseline_points and center_points):
        return {}

    kitchen_mid = kitchen_points[len(kitchen_points) // 2]
    baseline_mid = baseline_points[len(baseline_points) // 2]
    c_first, c_last = center_points[0], center_points[-1]
    # Orient the center line kitchen-end-first, baseline-end-last.
    if _dist(c_first, kitchen_mid) + _dist(c_last, baseline_mid) > _dist(c_last, kitchen_mid) + _dist(c_first, baseline_mid):
        center_points = list(reversed(center_points))
    center_x = center_points[0][0]

    kitchen_left, kitchen_right = _split_at_x(kitchen_points, center_x)
    baseline_left, baseline_right = _split_at_x(baseline_points, center_x)

    y_vals = [p[1] for p in kitchen_points + baseline_points]
    y_lo, y_hi = min(y_vals) - 20, max(y_vals) + 20
    y_lo, y_hi = min(y_lo, y_hi), max(y_lo, y_hi)

    zones = {}
    if left_side_points:
        side = sorted((p for p in left_side_points if y_lo <= p[1] <= y_hi), key=lambda p: p[1])
        if side:
            zones["left"] = kitchen_left + center_points + list(reversed(baseline_left)) + list(reversed(side))
    if right_side_points:
        side = sorted((p for p in right_side_points if y_lo <= p[1] <= y_hi), key=lambda p: p[1])
        if side:
            # Mirror of the left zone's winding: top edge center->right (kitchen_right
            # as split, already ascending), down the right sideline, bottom edge
            # right->center (baseline reversed), back up the center line to close.
            zones["right"] = kitchen_right + side + list(reversed(baseline_right)) + list(reversed(center_points))
    return zones


def load_calibration(path: str) -> dict:
    """Loads a calibration JSON saved by src.court_calibration_ui: each of the
    6 court parts (net, baseline, both sidelines, kitchen line, center line),
    manually traced (or explicitly skipped -- None -- when not visible in
    this camera's framing). Nothing here is derived/extrapolated: a skipped
    part is simply absent, never guessed.

    A line is a POLYLINE (as many clicked points as needed, to follow this
    camera's lens-distortion curve) and the net is a POLYGON (its 4-corner
    rectangle) -- both are expanded here into the flat list of straight
    2-point segments that draw_court_overlay() draws. This is the preferred,
    reliable source of court lines -- use it instead of detect_court_lines*()
    whenever a calibration file exists."""
    import json

    with open(path) as f:
        data = json.load(f)

    raw_points: dict[str, list] = {}
    lines = []
    net_left = net_right = None
    for name, feat in data["lines"].items():
        if feat is None:
            continue  # explicitly skipped -- not visible in this camera's framing
        pts = [(int(p[0]), int(p[1])) for p in feat["points"]]
        raw_points[name] = pts
        closed = feat.get("type") == "polygon"
        seq = pts + [pts[0]] if closed else pts
        for i in range(len(seq) - 1):
            lines.append((*seq[i], *seq[i + 1]))
        if name == "net" and len(pts) >= 2:
            net_left, net_right = pts[0], pts[1]  # top-left, top-right corners

    # Kitchen + service-box zones: close each polygon using already-traced,
    # trusted lines -- connecting real points, never inventing new ones. Only
    # built when every line a zone needs was actually traced (not skipped).
    zones: dict[str, list] = {}
    net_pts = raw_points.get("net")
    if net_pts and len(net_pts) == 4:
        _, _, net_br, net_bl = net_pts  # net's BASE (floor-level), shared by both halves
        if "kitchen_line" in raw_points:
            zones["near_kitchen"] = _build_kitchen_zone(
                net_bl, net_br, raw_points["kitchen_line"],
                raw_points.get("left_sideline"), raw_points.get("right_sideline"),
            )
        if "far_kitchen_line" in raw_points:
            zones["far_kitchen"] = _build_kitchen_zone(
                net_bl, net_br, raw_points["far_kitchen_line"],
                raw_points.get("far_left_sideline"), raw_points.get("far_right_sideline"),
            )

    for prefix, kitchen_key, baseline_key, center_key, left_key, right_key in (
        ("near", "kitchen_line", "baseline", "center_line", "left_sideline", "right_sideline"),
        ("far", "far_kitchen_line", "far_baseline", "far_center_line", "far_left_sideline", "far_right_sideline"),
    ):
        service_zones = _build_service_zones(
            raw_points.get(kitchen_key), raw_points.get(baseline_key), raw_points.get(center_key),
            raw_points.get(left_key), raw_points.get(right_key),
        )
        for side, pts in service_zones.items():
            zones[f"{prefix}_{side}_service"] = pts

    for zone_pts in zones.values():
        for i in range(len(zone_pts)):
            p1, p2 = zone_pts[i], zone_pts[(i + 1) % len(zone_pts)]
            lines.append((*p1, *p2))

    return {"lines": lines, "net_left": net_left, "net_right": net_right, "zones": zones}


def draw_court_overlay(frame, lines, profile: CourtProfile, color=(0, 230, 255), alpha: float = 0.65, thickness: int = 2,
                        draw_net_stub: bool = True):
    """Court-line overlay. `color` defaults to a vivid amber/yellow -- a thin,
    low-alpha white overlay (the original attempt) was nearly invisible
    against this court's own light-gray/blue surface. `draw_net_stub` draws
    the profile's fallback net-line-plus-posts; turn it off when `lines`
    already includes a calibrated net rectangle, to avoid drawing the net
    twice from two different (and possibly disagreeing) sources."""
    overlay = frame.copy()

    for x1, y1, x2, y2 in lines:
        cv2.line(overlay, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

    if draw_net_stub:
        net_left, net_right = profile.net_left_post, profile.net_right_post
        cv2.line(overlay, net_left, net_right, color, thickness, cv2.LINE_AA)
        for post in (net_left, net_right):
            cv2.line(overlay, post, (post[0], post[1] + profile.net_post_height), color, thickness, cv2.LINE_AA)

    return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)


# --- Top-down reconstruction (geometry sanity-check) ---------------------
#
# Finding the 4 corners by fitting a long line through a short Hough fragment
# and extrapolating it turned out unreliable: a tiny angle error, amplified
# over hundreds of pixels (and this camera's visible lens distortion -- the
# baseline is visibly curved, not straight), threw the far end off by 150px+.
# Measuring the court surface's own color boundary directly at a row close to
# the baseline is far more robust, since it doesn't extrapolate at all.

def find_court_corners(frame, profile: CourtProfile):
    """Net-left/net-right come from the profile's calibrated net posts. The two
    baseline corners are measured directly: the leftmost/rightmost pixel of the
    court's own playing-surface color at profile.baseline_row. Returns None if
    the court surface isn't found at that row (e.g. a player standing on it)."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower, upper = profile.court_surface_hsv
    mask = cv2.inRange(hsv, lower, upper)

    row = mask[profile.baseline_row]
    xs = np.where(row > 0)[0]
    if len(xs) == 0:
        return None

    base_left = (int(xs.min()), profile.baseline_row)
    base_right = (int(xs.max()), profile.baseline_row)
    return {
        "net_left": profile.net_left_post,
        "net_right": profile.net_right_post,
        "base_left": base_left,
        "base_right": base_right,
    }


def build_topdown_homography(corners: dict, court_width_m: float, half_depth_m: float, px_per_meter: float = 80.0):
    """Homography mapping the visible near half-court's 4 corners to a clean
    top-down rectangle sized from real-world court dimensions."""
    image_pts = np.array(
        [corners["net_left"], corners["net_right"], corners["base_right"], corners["base_left"]],
        dtype=np.float32,
    )
    w_px, d_px = court_width_m * px_per_meter, half_depth_m * px_per_meter
    world_pts = np.array([(0, 0), (w_px, 0), (w_px, d_px), (0, d_px)], dtype=np.float32)
    H = cv2.getPerspectiveTransform(image_pts, world_pts)
    return H, (int(w_px), int(d_px))
