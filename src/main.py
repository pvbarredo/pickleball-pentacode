import argparse
import os

from src.lap_shim import install as install_lap_shim

install_lap_shim()

import cv2
from ultralytics import YOLO

from src.config import Config
from src.court_roi import in_court
from src.draw import draw_pose, draw_tracked_boxes
from src.names import NameAssigner
from src.reid_gallery import ReIDGallery
from src.track_resolver import TrackResolver


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Detect and track persons on a pickleball court video.")
    cfg = Config()
    parser.add_argument("--input", default=cfg.input_path)
    parser.add_argument("--output", default=cfg.output_path)
    parser.add_argument("--model", default=cfg.model_name)
    parser.add_argument("--pose-model", default=cfg.pose_model_name)
    parser.add_argument("--conf", type=float, default=cfg.conf)
    parser.add_argument("--show", action="store_true", default=cfg.show)
    parser.add_argument("--no-show", dest="show", action="store_false")
    parser.add_argument("--max-frames", type=int, default=cfg.max_frames)
    parser.add_argument("--pose", action="store_true", default=cfg.pose, help="Also run pose estimation and draw skeletons")
    parser.add_argument("--kpt-conf", type=float, default=cfg.kpt_conf, help="Min keypoint confidence to draw")
    parser.add_argument("--tracker", default=cfg.tracker)
    parser.add_argument("--min-box-height", type=int, default=cfg.min_box_height, help="Ignore boxes shorter than this (px); filters distant/background people")
    parser.add_argument("--court-only", action="store_true", default=cfg.court_only, help="Restrict detection to the main court (ignores background courts/spectators)")
    parser.add_argument("--seed-gallery", default=cfg.seed_gallery, help="Path to a gallery.json from src.enroll to start with user-given names")
    parser.add_argument("--confirm-seconds", type=float, default=cfg.confirm_seconds, help="How long to wait, averaging appearance, before giving a new track a brand-new name instead of matching it to someone known")
    parser.add_argument("--match-thresh", type=float, default=cfg.match_thresh, help="Min appearance similarity to reuse a known name; higher = fewer false matches, more missed re-identifications")
    args = parser.parse_args()

    return Config(
        input_path=args.input,
        output_path=args.output,
        model_name=args.model,
        pose_model_name=args.pose_model,
        conf=args.conf,
        show=args.show,
        max_frames=args.max_frames,
        pose=args.pose,
        kpt_conf=args.kpt_conf,
        tracker=args.tracker,
        min_box_height=args.min_box_height,
        court_only=args.court_only,
        seed_gallery=args.seed_gallery,
        confirm_seconds=args.confirm_seconds,
        match_thresh=args.match_thresh,
    )


def main() -> None:
    cfg = parse_args()

    os.makedirs(os.path.dirname(cfg.output_path) or ".", exist_ok=True)

    model_name = cfg.pose_model_name if cfg.pose else cfg.model_name
    model = YOLO(model_name)

    cap = cv2.VideoCapture(cfg.input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {cfg.input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(cfg.output_path, fourcc, fps, (width, height))

    print(f"Input: {cfg.input_path} ({width}x{height} @ {fps:.1f}fps, {total_frames} frames)")
    print(f"Model: {model_name} | conf={cfg.conf} | tracker={cfg.tracker} | pose={cfg.pose}")

    frame_count = 0
    unique_ids = set()
    name_assigner = NameAssigner()
    gallery = ReIDGallery(match_thresh=cfg.match_thresh)
    if cfg.seed_gallery:
        gallery.load(cfg.seed_gallery)
        print(f"Seeded gallery with {len(gallery)} named player(s) from {cfg.seed_gallery}")
    resolver = TrackResolver(gallery, name_assigner, fps=fps, confirm_seconds=cfg.confirm_seconds)

    results_stream = model.track(
        source=cfg.input_path,
        classes=[cfg.person_class_id],
        conf=cfg.conf,
        persist=True,
        tracker=cfg.tracker,
        stream=True,
        verbose=False,
    )

    for result in results_stream:
        frame = result.orig_img
        boxes = result.boxes

        if boxes is not None and boxes.id is not None:
            xyxy = boxes.xyxy.cpu().numpy()
            track_ids = boxes.id.int().cpu().tolist()
            confs = boxes.conf.cpu().tolist()

            keep = [
                i for i, box in enumerate(xyxy)
                if (box[3] - box[1]) >= cfg.min_box_height and (not cfg.court_only or in_court(box))
            ]
            xyxy = xyxy[keep]
            track_ids = [track_ids[i] for i in keep]
            confs = [confs[i] for i in keep]

            unique_ids.update(track_ids)

            names = [resolver.resolve(tid, frame, box, frame_count) for tid, box in zip(track_ids, xyxy)]

            frame = draw_tracked_boxes(frame, xyxy, track_ids, confs, names)

            if cfg.pose and result.keypoints is not None:
                kpts_xy = result.keypoints.xy.cpu().numpy()[keep]
                kpts_conf = result.keypoints.conf.cpu().numpy()[keep]
                frame = draw_pose(frame, kpts_xy, kpts_conf, track_ids, cfg.kpt_conf)

        cv2.putText(
            frame, f"Frame {frame_count} | Players seen: {len(unique_ids)}",
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA,
        )

        writer.write(frame)

        if cfg.show:
            cv2.imshow("Pickleball Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Stopped by user (q pressed).")
                break

        frame_count += 1
        if cfg.max_frames and frame_count >= cfg.max_frames:
            break

    writer.release()
    if cfg.show:
        cv2.destroyAllWindows()

    unique_names = len(resolver.resolved_names())
    print(f"Done. Processed {frame_count} frames, {len(unique_ids)} tracker ID(s), {unique_names} re-identified player name(s).")
    print(f"Output written to: {cfg.output_path}")


if __name__ == "__main__":
    main()
