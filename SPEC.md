# Pickleball Player Detection & Tracking — Spec Plan

## Goal
Given `picklecam.mp4` (fixed overhead/sideline court camera, 332 MB), build a Python app that:
1. **Detects** all persons in each frame using YOLO.
2. **Draws bounding rectangles** around each detected person.
3. **Tracks** each person with a persistent ID from when they enter the frame to when they leave (or video ends), so the same player keeps the same ID across frames.
4. Outputs an annotated video for visual verification.

Out of scope for v1 (noted for later): mapping players to court positions (homography), shot/rally event detection, ball tracking, jersey/number ID, multi-camera sync.

## Environment check (done)
- No usable `python` on PATH — only the Windows Store alias stub. **Need to install a real Python 3.10/3.11** (python.org installer or `winget install Python.Python.3.11`) before anything else.
- GPU available: NVIDIA RTX 3060 Ti, 8GB VRAM — good for CUDA-accelerated inference (torch+cuda).
- No git repo initialized in `C:\Git\pickleball-yolo` yet.

## Tech stack
- **Python** 3.11
- **ultralytics** (YOLOv8) — provides both detection and a built-in multi-object tracker (ByteTrack/BoT-SORT) via `model.track(...)`, so we don't need a separate DeepSORT integration.
- **opencv-python** — video I/O, drawing boxes/IDs, optional live preview window.
- **torch** (CUDA build matching the 3060 Ti / installed CUDA driver) for GPU inference.
- Model weights: start with `yolov8n.pt` (fastest, good enough for person detection at this resolution) with `ym/yolov8m.pt` as a fallback if accuracy is insufficient. Detection filtered to COCO class `0` (person) only.

## Project structure
```
pickleball-yolo/
  SPEC.md
  requirements.txt
  .gitignore
  src/
    config.py          # paths, model name, conf/iou thresholds, tracker choice
    detect_track.py     # core: runs YOLO detect+track over video, yields per-frame results
    draw.py              # draws rectangles + track ID labels on frames
    main.py              # CLI entrypoint: python -m src.main --input picklecam.mp4
  output/
    picklecam_tracked.mp4   # annotated output video
    tracks.csv               # optional: frame, track_id, x1,y1,x2,y2, conf per row
  models/                 # downloaded YOLO weights (gitignored)
```

## Core approach
1. Load `yolov8n.pt` via `ultralytics.YOLO`.
2. Use `model.track(source=picklecam.mp4, classes=[0], persist=True, tracker="bytetrack.yaml", conf=0.4)` which runs detection + tracking frame-by-frame and returns stable `track_id`s per person.
3. For each frame: draw a rectangle per detected person box, label it with `ID {track_id}` (and confidence optionally), write frame to output video via `cv2.VideoWriter`.
4. Log per-frame track data (frame #, track_id, bbox, confidence) to `tracks.csv` for later analysis (e.g., dwell time, movement path).
5. CLI flags: `--input`, `--output`, `--model` (n/s/m), `--conf`, `--show` (live preview window), `--max-frames` (for quick testing on a short clip before running the full video).

## Milestones
1. **Setup**: install Python, create venv, `pip install ultralytics opencv-python`, verify `torch.cuda.is_available()` is True.
2. **Smoke test**: run detection+tracking on first ~300 frames (`--max-frames 300`) of `picklecam.mp4`, confirm boxes + consistent IDs appear on the 4 players.
3. **Full run**: process entire video, produce `output/picklecam_tracked.mp4` + `tracks.csv`.
4. **Review**: visually check output video — correct box placement, IDs stay consistent (no excessive ID switching when players cross/occlude near the net).
5. (Optional follow-up) Tune tracker params / try `bytetrack` vs `botsort`, or upgrade to `yolov8s`/`yolov8m` if small model misses players or boxes are jittery.

## Decisions
- Model: `yolov8n.pt` for first run.
- Output: annotated video only (`output/picklecam_tracked.mp4`).
- Live preview window: enabled (`--show`).
