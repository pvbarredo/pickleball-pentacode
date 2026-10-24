"""Diagnostic: trace exactly how tracker_ids get resolved to names in the
first N frames, so we can see fragmentation (same real player split into
multiple short-lived names) directly.
"""

import argparse

from src.lap_shim import install as install_lap_shim

install_lap_shim()

import cv2
from ultralytics import YOLO

from src.court_roi import in_court
from src.names import NameAssigner
from src.reid_gallery import ReIDGallery


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="picklecam.mp4")
    parser.add_argument("--max-frames", type=int, default=450)
    parser.add_argument("--min-box-height", type=int, default=100)
    parser.add_argument("--match-thresh", type=float, default=0.75)
    args = parser.parse_args()

    model = YOLO("yolov8n.pt")
    gallery = ReIDGallery(match_thresh=args.match_thresh)
    name_assigner = NameAssigner()
    track_names = {}
    track_first_seen = {}
    track_last_seen = {}
    track_frame_count = {}
    track_filtered_frames = {}  # frames where this tid existed but got filtered out

    results_stream = model.track(
        source=args.input, classes=[0], conf=0.4, persist=True,
        tracker="src/trackers/botsort_pickleball.yaml", stream=True, verbose=False,
    )

    frame_idx = 0
    for result in results_stream:
        frame = result.orig_img
        boxes = result.boxes
        if boxes is not None and boxes.id is not None:
            xyxy = boxes.xyxy.cpu().numpy()
            track_ids = boxes.id.int().cpu().tolist()
            for tid, box in zip(track_ids, xyxy):
                x1, y1, x2, y2 = box
                h = y2 - y1
                court_ok = in_court(box)
                height_ok = h >= args.min_box_height

                if not (height_ok and court_ok):
                    track_filtered_frames[tid] = track_filtered_frames.get(tid, 0) + 1
                    continue

                if tid not in track_names:
                    before = set(gallery._gallery)
                    name = gallery.identify(frame, box, frame_idx, name_assigner.new)
                    is_new = name not in before
                    track_names[tid] = name
                    print(f"[frame {frame_idx:4d}] tracker_id={tid:4d} -> {'NEW name' if is_new else 'MATCHED existing'} '{name}' (h={h:.0f})")
                else:
                    gallery.mark_active(track_names[tid], frame_idx)

                name = track_names[tid]
                track_first_seen.setdefault(tid, frame_idx)
                track_last_seen[tid] = frame_idx
                track_frame_count[tid] = track_frame_count.get(tid, 0) + 1

        frame_idx += 1
        if frame_idx >= args.max_frames:
            break

    print("\n--- summary per tracker_id ---")
    for tid in sorted(track_frame_count):
        name = track_names[tid]
        print(f"tracker_id={tid:4d} name={name:10s} first={track_first_seen[tid]:4d} last={track_last_seen[tid]:4d} "
              f"kept_frames={track_frame_count[tid]:4d} filtered_frames={track_filtered_frames.get(tid, 0):4d}")

    print("\n--- summary per name (merged) ---")
    name_totals = {}
    for tid, name in track_names.items():
        name_totals[name] = name_totals.get(name, 0) + track_frame_count[tid]
    for name, total in sorted(name_totals.items(), key=lambda kv: -kv[1]):
        tids = [tid for tid, n in track_names.items() if n == name]
        print(f"{name:10s} total_frames={total:4d} ({total/30:.1f}s) from tracker_id(s) {tids}")


if __name__ == "__main__":
    main()
