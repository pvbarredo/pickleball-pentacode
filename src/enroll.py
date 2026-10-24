"""Fast pass over the video that finds each distinct person and saves one
representative photo per person, so they can be named before the full
detect+track+pose run. Uses the same tracker + persistent re-ID gallery as
main.py (dense frame-to-frame tracking, not sparse sampling) so a single
real player doesn't get split into multiple photos.
"""

import argparse
import glob
import os
import time

from src.lap_shim import install as install_lap_shim

install_lap_shim()

import cv2
from PIL import Image
from ultralytics import YOLO

from src.court_roi import in_court
from src.names import NameAssigner
from src.reid_gallery import ReIDGallery


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan a video and collect one photo per distinct person.")
    parser.add_argument("--input", default="picklecam.mp4")
    parser.add_argument("--out-dir", default="output/enroll")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--tracker", default="src/trackers/botsort_pickleball.yaml")
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--match-thresh", type=float, default=0.75)
    parser.add_argument("--min-box-height", type=int, default=100)
    parser.add_argument("--min-seconds", type=float, default=2.0, help="Only keep people detected for at least this many seconds total")
    parser.add_argument("--gif-seconds", type=float, default=1.5, help="Length of the ID clip (GIF) captured from each person's first appearance")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit the scan to the first N frames (fast preview of a portion of the video)")
    parser.add_argument("--imgsz", type=int, default=480, help="Inference resolution; smaller is faster for the enrollment scan")
    parser.add_argument("--court-only", action="store_true", default=True)
    parser.add_argument("--no-court-only", dest="court_only", action="store_false")
    parser.add_argument("--reuse", action="store_true", help="Keep any existing photos/mapping in --out-dir instead of starting a fresh scan")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if not args.reuse:
        # Every scan is for ONE video upload and should never silently carry over a
        # previous video's enrolled faces/names -- start clean unless told otherwise.
        stale = (
            glob.glob(os.path.join(args.out_dir, "*.jpg"))
            + glob.glob(os.path.join(args.out_dir, "*.gif"))
            + glob.glob(os.path.join(args.out_dir, "*.json"))
        )
        for f in stale:
            os.remove(f)
        if stale:
            print(f"Cleared {len(stale)} file(s) from a previous enrollment in {args.out_dir}.")

    model = YOLO(args.model)

    cap = cv2.VideoCapture(args.input)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    min_frames = round(fps * args.min_seconds)
    gif_max_frames = max(2, round(fps * args.gif_seconds))
    GIF_THUMB = (160, 220)

    gallery = ReIDGallery(match_thresh=args.match_thresh)
    name_assigner = NameAssigner()
    track_names: dict[int, str] = {}
    crop_paths: dict[str, str] = {}
    first_seen_frame: dict[str, int] = {}
    active_frames: dict[str, int] = {}
    gif_frames: dict[str, list] = {}

    results_stream = model.track(
        source=args.input,
        classes=[0],
        conf=args.conf,
        imgsz=args.imgsz,
        persist=True,
        tracker=args.tracker,
        stream=True,
        verbose=False,
    )

    start_time = time.time()

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
                if h < args.min_box_height:
                    continue
                if args.court_only and not in_court(box):
                    continue

                if tid not in track_names:
                    track_names[tid] = gallery.identify(frame, box, frame_idx, name_assigner.new)
                else:
                    gallery.mark_active(track_names[tid], frame_idx)
                name = track_names[tid]
                active_frames[name] = active_frames.get(name, 0) + 1

                # Capture the ID photo at this person's FIRST appearance in the video --
                # that's the reference we need in order to recognize them from start to
                # finish, not whatever frame happens to look best later on.
                if name not in crop_paths:
                    first_seen_frame[name] = frame_idx
                    cx1, cy1, cx2, cy2 = [int(v) for v in box]
                    crop = frame[max(cy1, 0):cy2, max(cx1, 0):cx2]
                    path = os.path.join(args.out_dir, f"{name}.jpg")
                    cv2.imwrite(path, crop)
                    crop_paths[name] = path

                # Keep collecting frames (right from their first appearance) for a short
                # ID clip -- a moving clip makes it much easier to recognize someone than
                # one static photo.
                buf = gif_frames.setdefault(name, [])
                if len(buf) < gif_max_frames:
                    cx1, cy1, cx2, cy2 = [int(v) for v in box]
                    crop = frame[max(cy1, 0):cy2, max(cx1, 0):cx2]
                    if crop.size:
                        buf.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

        frame_idx += 1
        if args.max_frames and frame_idx >= args.max_frames:
            break

    elapsed = time.time() - start_time

    kept = {name: path for name, path in crop_paths.items() if active_frames.get(name, 0) >= min_frames}
    dropped = set(crop_paths) - set(kept)
    for name in dropped:
        try:
            os.remove(crop_paths[name])
        except OSError:
            pass

    gallery.save(os.path.join(args.out_dir, "gallery.json"))

    for name in kept:
        frames = gif_frames.get(name, [])
        if len(frames) < 2:
            continue
        pil_frames = [Image.fromarray(f).resize(GIF_THUMB) for f in frames]
        gif_path = os.path.join(args.out_dir, f"{name}.gif")
        pil_frames[0].save(gif_path, save_all=True, append_images=pil_frames[1:], duration=80, loop=0)

    print(f"Scanned {frame_idx} frame(s) in {elapsed:.1f}s ({frame_idx / max(elapsed, 0.001):.1f} fps).")
    print(f"Found {len(crop_paths)} candidate person(s), kept {len(kept)} seen for >= {args.min_seconds:.0f}s "
          f"(dropped {len(dropped)} brief/spurious one(s)):")
    for name, path in kept.items():
        print(f"  {name}: first seen at {first_seen_frame[name] / fps:.1f}s, "
              f"on screen {active_frames[name] / fps:.1f}s total -> {path}")


if __name__ == "__main__":
    main()
