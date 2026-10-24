import argparse
import os

from src.lap_shim import install as install_lap_shim

install_lap_shim()

import cv2
from ultralytics import YOLO

from src.ball_tracker import BallTracker
from src.config import Config
from src.court_lines import detect_court_lines_stable, draw_court_overlay, load_calibration
from src.court_profiles import get_profile
from src.court_roi import in_court
from src.draw import draw_pose, draw_tracked_boxes
from src.minimap import Minimap
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
    parser.add_argument("--detect-lines", action="store_true", default=cfg.detect_lines, help="Overlay detected court lines (sidelines, kitchen, net)")
    parser.add_argument("--court-profile", default=cfg.court_profile, help="Camera-angle calibration profile (see src/court_profiles.py)")
    parser.add_argument("--court-calibration", default=cfg.court_calibration, help="Path to a calibration JSON from src.court_calibration_ui (preferred over auto-detection when given)")
    parser.add_argument("--track-ball", action="store_true", default=cfg.track_ball, help="Detect the ball and draw a fading dashed trail of its recent path")
    parser.add_argument("--ball-conf", type=float, default=cfg.ball_conf)
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
        detect_lines=args.detect_lines,
        court_profile=args.court_profile,
        court_calibration=args.court_calibration,
        track_ball=args.track_ball,
        ball_conf=args.ball_conf,
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

    court_lines = []
    zones = {}
    draw_net_stub = True  # fallback net-line-plus-posts from the profile; off once calibration supplies its own net
    court_profile = get_profile(cfg.court_profile)
    if cfg.detect_lines:
        if cfg.court_calibration and os.path.exists(cfg.court_calibration):
            calib = load_calibration(cfg.court_calibration)
            court_lines = calib["lines"]
            zones = calib["zones"]
            draw_net_stub = not (calib["net_left"] and calib["net_right"])
            print(f"Loaded {len(court_lines)} calibrated court line(s) and {len(zones)} zone(s) from {cfg.court_calibration}.")
        else:
            court_lines = detect_court_lines_stable(cfg.input_path, court_profile)
            print(f"Detected {len(court_lines)} court line segment(s) using profile '{court_profile.name}' (auto, no calibration file found).")

    # Built whenever zones exist, regardless of --show: shade_zones() (the
    # always-on per-zone color) has nothing to do with the mouse and belongs
    # in the written file too. Only the hover/panel/mouse-debug bits below
    # need an actual interactive window.
    minimap = Minimap(zones, (height, width)) if zones else None
    mouse_pos = [None]
    if cfg.show and minimap is not None:
        cv2.namedWindow("Pickleball Tracking")
        cv2.setMouseCallback("Pickleball Tracking", lambda e, x, y, f, p: mouse_pos.__setitem__(0, (x, y)))

    ball_model = None
    ball_tracker = None
    if cfg.track_ball:
        # Always a SEPARATE model instance: `model` is mid-iteration through its own
        # track(stream=True) generator, and calling predict() on that same object
        # corrupts its internal predictor state and hangs the whole run.
        ball_model = YOLO(cfg.model_name)
        ball_tracker = BallTracker(fps=fps, conf=cfg.ball_conf, imgsz=cfg.ball_imgsz)

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

        if court_lines or cfg.detect_lines:
            frame = draw_court_overlay(frame, court_lines, court_profile, draw_net_stub=draw_net_stub)

        if minimap is not None:
            # Always-on per-zone color shade (which part is which) isn't
            # mouse-dependent, so it's baked into the real output video too --
            # only the hover boost and the minimap panel itself are preview-only.
            frame = minimap.shade_zones(frame)

        if cfg.track_ball:
            ball_tracker.detect(ball_model, frame, frame_count)
            frame = ball_tracker.draw(frame, frame_count)

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
            # The hover boost, the minimap panel itself, and the mouse-position
            # debug marker are all mouse-dependent -- interactive-only, applied
            # to a display copy, never baked into the frame written to disk.
            display_frame = frame
            if minimap is not None:
                hovered = minimap.hovered_zone(mouse_pos[0])
                display_frame = minimap.hover_boost(frame.copy(), hovered)
                display_frame = minimap.draw(display_frame, hovered)
                display_frame = minimap.draw_mouse_debug(display_frame, mouse_pos[0])
            cv2.imshow("Pickleball Tracking", display_frame)
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
