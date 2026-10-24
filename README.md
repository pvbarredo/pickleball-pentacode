# Pickleball Player Detection, Tracking & Re-Identification

Detects every person on a pickleball court in a fixed-camera video, tracks
each one frame-to-frame, and (optionally) lets you **name** specific players
from a short visual enrollment pass so the final output video shows their
real name wherever they appear — including after they leave the frame and
come back, or after the tracker's own ID switches.

Also draws a full-body pose skeleton (17 COCO keypoints) on top of each
tracked player.

## How it works (detection → ID, end to end)

The system is three layers stacked on top of each other. Each layer exists
because the one below it isn't enough on its own.

### 1. Detection — "where are the people?"

[`YOLO`](https://github.com/ultralytics/ultralytics) (`yolov8n.pt`, or
`yolov8n-pose.pt` when pose is enabled) runs on every frame and returns a
bounding box for every person (COCO class `0`) above a confidence threshold
(`--conf`, default `0.4`).

### 2. Tracking — "is this the same box as last frame?"

[BoT-SORT](https://github.com/NirAharon/BoT-SORT) (via Ultralytics'
built-in tracker, config in [`src/trackers/botsort_pickleball.yaml`](src/trackers/botsort_pickleball.yaml))
assigns a numeric `tracker_id` to each detection and carries it forward
frame-to-frame using motion + IoU + appearance. Tuned for this use case:

- `track_buffer: 150` (~5s @ 30fps) — survives a player being occluded at
  the net or briefly stepping out of frame.
- `new_track_thresh: 0.6` — raised so noisy/partial detections don't spawn
  spurious new IDs as easily.
- `with_reid: True` — the tracker's own lightweight appearance check, on
  top of motion/IoU.

**Tracking alone is not enough**: if a track is lost for longer than the
buffer (someone walks off for a water break, or the tracker's ID switches
on a collision near the net), the person comes back as a **brand-new**
`tracker_id` with no memory of who they were. That's what layer 3 fixes.

Two **per-frame filters** run before a detection counts toward anyone:

- `--min-box-height` — drops small/distant detections (background courts,
  spectators behind the fence).
- `--court-only` + [`src/court_roi.py`](src/court_roi.py) — a polygon drawn
  over the featured court; a detection only counts if its foot-point (the
  bottom-center of the box) falls inside it. This is **tuned for this
  specific camera angle** (`picklecam.mp4`, 1920×1080) — redraw the polygon
  if you point this at a different camera.

### 3. Persistent re-identification — "have I seen this person before?"

This is the part that survives a `tracker_id` being lost and reissued.
[`src/reid_gallery.py`](src/reid_gallery.py) keeps, per named person, a
small **list of reference appearance signatures** (HSV color histograms of
their torso region — middle 60%×70% of the box, which skips the head and
shoes and trims background at the edges).

Important design choice: **we never blend multiple samples into one
average histogram.** An early version did, and it was the main source of
mis-identification — averaging ten photos of the same shirt under
different gym lighting produces a "mushy" composite that doesn't closely
resemble *any* of the real instances. Instead, every enrollment clip (minus
obvious outliers — a badly-lit or mostly-background crop) is kept as its
own exemplar, and a new detection is matched against the single
**best-scoring** exemplar (`cv2.compareHist`, `HISTCMP_CORREL`), not an
average.

When a brand-new `tracker_id` shows up, [`src/track_resolver.py`](src/track_resolver.py)
doesn't commit to a name off one possibly-bad frame. It waits up to
`--confirm-seconds` (default `1.0`), re-checking **each individual frame**
(not an average of them — same reasoning as above) against the gallery. The
box is briefly labeled `"..."` while this happens. Two outcomes:

- A frame scores ≥ `--match-thresh` (default `0.75`) against a known
  person → that `tracker_id` is now that person, permanently, and the
  matching frame is added as a new exemplar (the gallery keeps learning).
- Nothing matches within the window → a fresh name is minted, seeded with
  every frame buffered during that window as its first exemplars.

A name can't "steal" another person who is currently active on screen
(`active_window`): a name that was seen in the last 5 frames under a
*different* `tracker_id` is excluded from matching, since they're clearly
still in frame under their own ID.

## The enrollment / naming workflow

Random placeholder names (`Ace`, `Nova`, `Kenneth`-length names from
[`src/names.py`](src/names.py)) aren't useful on their own — you want *your*
names. The workflow:

1. **`src/enroll.py`** — a fast, no-pose, no-video-output pass over the
   *entire* video (dense BoT-SORT tracking + the same re-ID gallery logic
   above), collecting:
   - one **first-appearance photo** per distinct identity (captured the
     *moment* they're first seen, not some "best" frame later — this is
     what lets you ID someone who's only in the first few seconds),
   - a short **animated GIF** clip (1.5s) from the same window, for easier
     visual recognition than a single static photo,
   - only keeping identities seen for at least `--min-seconds` (default
     `2.0`) total, to drop one-off background passersby.

   Every run **starts fresh** — it clears any photos/GIFs/mapping left over
   from a previous video so you never accidentally reuse a different
   video's enrollment (pass `--reuse` to opt out).

2. **`src/label_ui.py`** — opens a Tkinter window with every candidate as
   an animated thumbnail. Check the ones that are the same real person,
   type one name, click "Assign to selected" (repeat per person); check and
   "Discard selected" for anything that isn't a player you care about.
   "Done - Save" writes `output/enroll/name_mapping.json`.

3. **`src/apply_naming.py`** — merges the placeholders you grouped under
   each name into that name's exemplar list (dropping any exemplar whose
   histogram looks nothing like the group's median — the "one bad crop"
   problem from above), and writes `output/enroll/gallery_named.json`.

4. **`src/main.py --seed-gallery output/enroll/gallery_named.json`** — the
   full run, now starting with your named players already in the gallery.

## Quick start

```bash
# 1. Scan the whole video and collect candidate photos/clips (fast, no pose/video output)
python -m src.enroll --input picklecam.mp4

# 2. Name them visually (opens a window)
python -m src.label_ui

# 3. Bake your names into the seed gallery
python -m src.apply_naming

# 4. Run the full video with your names, pose skeletons, and a live preview window
python -m src.main --input picklecam.mp4 --output output/tracked.mp4 \
  --pose --show --seed-gallery output/enroll/gallery_named.json --court-only
```

Press `q` in the preview window to stop early (whatever was processed so
far is still saved to `--output`).

### Useful flags on `src.main`

| Flag | Default | What it does |
|---|---|---|
| `--pose` | off | Also run pose estimation and draw the skeleton |
| `--show` / `--no-show` | on | Live OpenCV preview window while processing |
| `--seed-gallery <path>` | none | Start with named players from `apply_naming.py` |
| `--court-only` | off | Restrict detection to the main court (see `src/court_roi.py`) |
| `--min-box-height` | `0` | Ignore boxes shorter than this many px |
| `--match-thresh` | `0.75` | Min similarity to reuse a known name (higher = fewer false matches, more missed re-IDs) |
| `--confirm-seconds` | `1.0` | How long a new track gets to match an existing name before becoming a new one |
| `--tracker <path>` | `src/trackers/botsort_pickleball.yaml` | Tracker config |
| `--max-frames N` | none | Stop after N frames (quick tests) |

## Requirements

- **Python 3.11** (tested on 3.11.9; anything 3.10+ should work)
- **Windows** (tested) — should run on Linux/macOS too, aside from the
  `lap`-package DLL note below, which is Windows-specific
- **NVIDIA GPU + CUDA, strongly recommended but optional.** This project
  was built/tested on an RTX 3060 Ti (8GB). Everything falls back to CPU
  automatically if no CUDA GPU is available — detection/tracking will just
  be much slower (the full-video enrollment scan ran at ~50 fps on the
  3060 Ti; CPU-only would be a fraction of that).

### Libraries used

| Library | Used for |
|---|---|
| [`ultralytics`](https://pypi.org/project/ultralytics/) | YOLOv8 detection + pose models, BoT-SORT tracker |
| [`torch`](https://pytorch.org/) / `torchvision` | Model inference backend (GPU acceleration) |
| [`opencv-python`](https://pypi.org/project/opencv-python/) | Video I/O, drawing, HSV color histograms for re-ID |
| [`scipy`](https://pypi.org/project/scipy/) | Linear assignment solver (see `lap` note below) |
| [`Pillow`](https://pypi.org/project/Pillow/) | GIF encoding, thumbnails in the naming window |
| `tkinter` (stdlib) | The visual naming window |
| `numpy` | Everywhere (pulled in by the above) |

### How the GPU is used

`torch` is installed with CUDA support, and Ultralytics automatically runs
every YOLO forward pass (detection and pose) on the GPU when
`torch.cuda.is_available()` is `True` — there's no flag to set. Check it
worked with:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Setup on a new PC (cloning this repo)

```bash
git clone <this-repo-url>
cd pickleball-yolo

python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS

# Install CUDA-enabled PyTorch FIRST (plain `pip install torch` gives you a
# CPU-only build). Match the index URL to your CUDA version if different
# from cu121 -- see https://pytorch.org/get-started/locally/
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

YOLO model weights (`yolov8n.pt`, `yolov8n-pose.pt`) are **not** committed
to the repo — Ultralytics downloads them automatically the first time
they're needed (a few MB each).

Drop your own video in as `picklecam.mp4` (or pass `--input your-video.mp4`
to every command above), and run the Quick Start steps.

### A Windows-specific gotcha: the `lap` package

Ultralytics' tracker imports the `lap` package (a compiled linear-assignment
solver) at load time. On a machine with Windows Defender Application
Control / Smart App Control enabled, the freshly pip-installed `lap` DLL
can get silently blocked (`ImportError: DLL load failed ... blocked by an
Application Control policy`), which crashes tracking entirely.

[`src/lap_shim.py`](src/lap_shim.py) works around this by registering a
pure-Python/SciPy drop-in for `lap` in `sys.modules` *before* Ultralytics
ever imports the real one — every entry point (`main.py`, `enroll.py`,
`diag_live.py`) calls `install_lap_shim()` first thing. If your machine
doesn't hit this policy, the shim is a harmless no-op; you don't need to
remove it.

## Project structure

```
src/
  main.py            Full run: detect + track + re-ID + (optional) pose, writes output video
  enroll.py           Fast full-video scan: collects named-candidate photos/GIFs
  label_ui.py          Tkinter window for visually naming enrollment candidates
  apply_naming.py       Merges your naming choices into the final seed gallery
  reid_gallery.py        Persistent multi-exemplar appearance matching
  track_resolver.py       Delays new-track naming to retry matches per-frame
  court_roi.py              Main-court polygon filter (camera-specific)
  draw.py                    Box/label/skeleton drawing
  names.py                    Random name pool for un-enrolled players
  config.py                    CLI defaults
  lap_shim.py                   Windows DLL-block workaround (see above)
  trackers/botsort_pickleball.yaml   Tuned BoT-SORT config
output/                  Generated videos, enrollment photos/GIFs, galleries (gitignored)
```

## Known limitations

- Re-identification uses simple HSV color histograms, not a deep
  learning embedding — it's fast and dependency-light, but two players in
  similarly-colored shirts can be confused, and it's sensitive to venue
  lighting changes.
- `src/court_roi.py`'s polygon is hand-tuned for `picklecam.mp4`'s exact
  camera position; a different camera angle needs a new polygon (see the
  frame-0 annotated debug approach used during development — draw the
  polygon over a sample frame and adjust the four corner points).
