"""One-off diagnostic: detects court lines, finds the near half-court's 4
corners, and renders a top-down (bird's-eye) reconstruction with a reference
grid overlay -- so you can visually confirm the detected geometry really
looks like a pickleball court, compared against real-world proportions.
"""

import argparse
import os

import cv2
import numpy as np

from src.court_lines import build_topdown_homography, detect_court_lines_stable, draw_court_overlay, find_court_corners
from src.court_profiles import COURT_WIDTH_M, HALF_COURT_DEPTH_M, KITCHEN_DEPTH_M, get_profile


def draw_reference_grid(topdown, px_per_meter: float):
    """Draws the EXPECTED court markings (kitchen line, sidelines) in green, to
    compare against the real white lines visible in the warped frame underneath."""
    h, w = topdown.shape[:2]
    color = (0, 255, 0)
    cv2.rectangle(topdown, (0, 0), (w - 1, h - 1), color, 2)
    kitchen_y = int(KITCHEN_DEPTH_M * px_per_meter)
    cv2.line(topdown, (0, kitchen_y), (w, kitchen_y), color, 2)
    cv2.putText(topdown, "kitchen line (expected)", (10, kitchen_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return topdown


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="picklecam.mp4")
    parser.add_argument("--court-profile", default="picklecam_vertical")
    parser.add_argument("--out-dir", default="output/diag")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    profile = get_profile(args.court_profile)

    lines = detect_court_lines_stable(args.input, profile)
    print(f"Detected {len(lines)} merged line(s).")

    cap = cv2.VideoCapture(args.input)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 500)
    ok, frame = cap.read()
    cap.release()

    annotated = draw_court_overlay(frame, lines, profile)
    annotated_path = os.path.join(args.out_dir, "court_lines_annotated.jpg")
    cv2.imwrite(annotated_path, annotated)
    print(f"Saved annotated frame: {annotated_path}")

    corners = find_court_corners(frame, profile)
    if corners is None:
        print("Could not find all 4 corners (sideline/baseline detection failed) -- no top-down image produced.")
        return

    for name, pt in corners.items():
        cv2.circle(annotated, (int(pt[0]), int(pt[1])), 8, (0, 0, 255), -1)
        cv2.putText(annotated, name, (int(pt[0]) + 10, int(pt[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imwrite(os.path.join(args.out_dir, "court_corners.jpg"), annotated)

    px_per_meter = 80.0
    H, (w, d) = build_topdown_homography(corners, COURT_WIDTH_M, HALF_COURT_DEPTH_M, px_per_meter)
    topdown = cv2.warpPerspective(frame, H, (w, d))
    topdown = draw_reference_grid(topdown, px_per_meter)

    topdown_path = os.path.join(args.out_dir, "court_topdown.jpg")
    cv2.imwrite(topdown_path, topdown)
    print(f"Saved top-down reconstruction: {topdown_path} ({w}x{d}px, {px_per_meter}px/m)")


if __name__ == "__main__":
    main()
