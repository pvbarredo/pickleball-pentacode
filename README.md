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

## Court lines, kitchen/service zones, ball tracking, and the minimap

A separate feature set overlays the court geometry itself (lines, the net,
the kitchen/service-box zones) and tracks the ball. None of this touches
detection/tracking/re-ID above — it's purely visual/geometric.

### Why calibration is fully manual, not auto-detected

The first attempt detected lines automatically (HSV threshold for painted
white lines → Canny edges → `cv2.HoughLinesP` → cluster nearby segments into
one line per real edge). It worked, but this camera's visible lens
distortion (the baseline is visibly curved, not straight) kept breaking two
things that needed to be reliable:

- A 4-corner homography (click/detect the net posts + baseline corners,
  derive every other line by projecting known court dimensions through the
  resulting perspective transform) assumes straight lines. A tiny angle
  error in a short detected segment, extrapolated across hundreds of pixels
  *and* real lens distortion, threw the far end off by 150px+.
- The camera doesn't necessarily frame the whole court (a line can run off
  the edge of frame, or behind equipment) — an algorithm that always
  produces 4 corners can't represent "this line isn't visible here."

So calibration ([`src/court_calibration_ui.py`](src/court_calibration_ui.py))
is fully manual instead, and nothing is derived or extrapolated between
parts — every part you skip is simply absent, never guessed:

1. **Person-free background plate** ([`src/background_plate.py`](src/background_plate.py)):
   the per-pixel **median** across ~40 frames spread across the video. The
   camera is fixed, so the court is identical in every frame, but any given
   pixel is only covered by a moving player some of the time — across
   enough samples the median at each pixel is the real court surface, not a
   person. (Deliberately not generative AI: real frames already show every
   part of the court unoccluded at some point, so this is both more
   accurate — no hallucinated detail — and instant/free.)

2. **Click-trace each court part** on that background image, one at a time,
   with a side panel showing a schematic (which part is meant) and a
   zoomed-in hint crop (roughly where, in this specific photo) for each
   step:
   - The **5 lines** (near/far baseline, both sidelines, both kitchen
     lines, both center lines — 10 total across both halves) are each a
     **polyline**: click as many points as needed to follow the curve, then
     "Finish this line". A straight 2-point line cuts the corner on a
     curved one.
   - The **net** is a **4-corner polygon** (top-left, top-right,
     bottom-right, bottom-left of its visible face) — it has real height in
     the image, not just a top cable.
   - **"Skip this line"** on anything not visible in this camera's framing
     (e.g. the far baseline, often hidden behind equipment).
   - **"Redo a specific part"** re-traces just one already-finished part in
     place, without touching anything traced before or after it.
   - The window scales to fit the screen (capped at 1080p, never upscaled)
     and converts clicks back to the source image's own coordinate space,
     so calibration stays pixel-accurate regardless of display scaling.
   - Re-running the tool **resumes** your saved calibration instead of
     making you re-trace everything.

   Saves `output/calibration/picklecam_vertical.json`: each part's traced
   points, or `null` if skipped.

3. **`src/court_lines.py`** loads that file and expands each polyline/polygon
   into the flat list of straight segments `draw_court_overlay()` draws —
   no detection or derivation at video-processing time, just what you
   traced.

Calibration is **camera-specific** (what's hand-clicked: the polygon that
restricts detection in `src/court_roi.py`, the net's fallback position in
`src/court_profiles.py`), kept isolated from the **generic** algorithm code
(`src/court_lines.py`'s clustering/drawing math), so a different camera
angle — e.g. one looking across the court's *width* instead of down its
*length* — just needs a new `CourtProfile` and a fresh calibration JSON, no
logic changes. `src/court_lines_preview.py` is a leftover diagnostic from
the earlier auto-detection attempt (renders a top-down reconstruction to
sanity-check geometry); not part of the normal workflow anymore.

### Kitchen and service-box zones

Rather than re-clicking more rectangles, the 2 kitchen zones and 4
service-box zones are **built automatically from the lines you already
traced** (`_build_kitchen_zone` / `_build_service_zones` in
`src/court_lines.py`) — connecting real traced points, never inventing new
geometry:

- **Kitchen zone**: the net's own base (where it meets the floor — *not*
  its top cable; that's elevated by the net's height and irrelevant to a
  floor-level zone) on one side, the kitchen line's traced curve on the
  other. The left/right edges use the **actual merge point**: wherever that
  sideline's own traced curve comes closest to the net's base line — not
  the net polygon's literal clicked corner, which may not land exactly on
  the true sideline and left a visible gap/seam near the posts.
- **Service boxes**: closed from the kitchen line + baseline + center line
  + the relevant sideline, split left/right at the center line.

Both zones always show a **colored shade** (`src/minimap.py`'s
`shade_zones()`), baked into the actual output video — not mouse-dependent,
so it's there whether or not `--show` is used.

### Ball tracking

[`src/ball_tracker.py`](src/ball_tracker.py) detects the pickleball (COCO
`sports ball`, class 32) and draws a fading, dashed trail of its recent
flight. Two things this needed that the person-detection pass doesn't:

- **A separate, higher-resolution inference pass.** The ball is too small
  for YOLO's default 640px inference size — it was invisible in testing.
  `--ball-imgsz` (default `1280`) fixes that, at real extra cost (another
  full forward pass, at ~4x the pixel area of the person pass).
- **Its own model instance.** `model` is mid-iteration through its own
  `track(stream=True)` generator; calling `predict()` on that *same* object
  for the ball pass corrupted its internal state and hung the whole
  process. The ball model is always a second, independent `YOLO(...)`.

The trail only connects two detections if they're within `--max-gap`-ish
consecutive frames of each other (hardcoded ~0.2s) — otherwise it starts a
new disconnected stroke rather than drawing a false "bridge" line across a
gap where the ball wasn't detected (briefly occluded, or genuinely a new
flight).

### The minimap (hover-highlight)

In the live `--show` window only, a small panel (top-right) shows every
zone in miniature; hovering a zone there brightens that same zone on the
main video (`src/minimap.py`'s `Minimap` class, `hover_boost()` +
`draw()`). This is strictly mouse-dependent and only makes sense
interactively — never baked into a written file. If hovering doesn't seem
to register, a small red crosshair (`draw_mouse_debug()`) shows exactly
where the app thinks your mouse is; if it doesn't track your real cursor,
that's a coordinate-scaling mismatch with the raw OpenCV window (seen on
Windows with display scaling), not a logic bug.

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

To also get court lines, kitchen/service zones, and the ball trail:

```bash
# One-time: generate the background plate, then click-trace the court
python -m src.background_plate
python -m src.court_calibration_ui

# Then any src.main run with --detect-lines picks up output/calibration/picklecam_vertical.json automatically
python -m src.main --input picklecam.mp4 --output output/tracked.mp4 \
  --show --detect-lines --track-ball
```

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
| `--detect-lines` | off | Overlay court lines + kitchen/service zone shading (see calibration section above) |
| `--court-calibration <path>` | `output/calibration/picklecam_vertical.json` | Calibration JSON from `src.court_calibration_ui` |
| `--court-profile` | `picklecam_vertical` | Camera-angle profile (see `src/court_profiles.py`) |
| `--track-ball` | off | Detect the ball and draw its fading trail |
| `--ball-conf` | `0.3` | Min confidence for a ball detection |

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
| [`Pillow`](https://pypi.org/project/Pillow/) | GIF encoding, thumbnails in the naming/calibration windows |
| `tkinter` (stdlib) | The visual naming and court-calibration windows |
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
  main.py                  Full run: detect + track + re-ID + (optional) pose/lines/ball, writes output video
  enroll.py                 Fast full-video scan: collects named-candidate photos/GIFs
  label_ui.py                 Tkinter window for visually naming enrollment candidates
  apply_naming.py                Merges your naming choices into the final seed gallery
  reid_gallery.py                  Persistent multi-exemplar appearance matching
  track_resolver.py                   Delays new-track naming to retry matches per-frame
  court_roi.py                          Main-court polygon filter (player detection, camera-specific)
  draw.py                                 Box/label/skeleton drawing
  names.py                                  Random name pool for un-enrolled players
  config.py                                   CLI defaults
  lap_shim.py                                   Windows DLL-block workaround (see above)
  trackers/botsort_pickleball.yaml                Tuned BoT-SORT config
  background_plate.py      Person-free court photo via median-of-frames
  court_calibration_ui.py   Tkinter window: click-trace every court line + the net
  court_profiles.py          Per-camera-angle calibration (net fallback position, search ROI)
  court_lines.py               Loads a calibration JSON; builds kitchen/service zones; draws overlay
  court_lines_preview.py        Leftover diagnostic from the earlier auto-detection attempt
  ball_tracker.py          Ball detection (separate higher-res model pass) + fading trail
  minimap.py                Always-on zone shading + the live hover-highlight panel
output/                  Generated videos, enrollment photos/GIFs, galleries, calibration JSON (gitignored)
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
- Court-line calibration (`output/calibration/picklecam_vertical.json`) is
  per-video/per-camera-position data, not committed to the repo (it's under
  the gitignored `output/`) — a new video needs its own
  `src.background_plate` + `src.court_calibration_ui` pass.
- The far side of the court (beyond the net) is only as complete as what's
  actually visible and traced — far-baseline is commonly skipped where
  equipment/staging blocks the camera's view of it.
