"""Replays main.py's exact resolve logic on the first N frames with the real
seed gallery, printing the match score behind every decision, so we can see
exactly why a track got (or didn't get) matched to a known name."""

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
    parser.add_argument("--seed-gallery", default="output/enroll/gallery_named.json")
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--min-box-height", type=int, default=100)
    parser.add_argument("--match-thresh", type=float, default=0.75)
    args = parser.parse_args()

    model = YOLO("yolov8n-pose.pt")
    gallery = ReIDGallery(match_thresh=args.match_thresh)
    gallery.load(args.seed_gallery)
    name_assigner = NameAssigner()
    track_names = {}

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
                if h < args.min_box_height or not in_court(box):
                    continue
                if tid not in track_names:
                    hist = gallery._histogram(gallery.crop_box(frame, box))
                    if hist is not None:
                        best_name, best_score = gallery.best_match(hist, frame_idx)
                        print(f"[frame {frame_idx:4d}] NEW tracker_id={tid:4d} best_match={best_name} score={best_score:.3f} "
                              f"-> {'MATCHED' if best_score >= args.match_thresh else 'rejected -> new name'}")
                    name = gallery.identify(frame, box, frame_idx, name_assigner.new)
                    track_names[tid] = name
                    print(f"              resolved name = {name}")
                else:
                    gallery.mark_active(track_names[tid], frame_idx)
        frame_idx += 1
        if frame_idx >= args.max_frames:
            break


if __name__ == "__main__":
    main()
