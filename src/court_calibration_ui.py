"""Visual court calibration: trace each court line directly on a person-free
background plate -- net, baseline, both sidelines, the kitchen line, and the
center line.

Fully manual, with two things this camera's lens distortion and the net's
real depth in the image both require:
  - Every LINE (baseline, sidelines, kitchen line, center line) is a
    POLYLINE of as many points as you click, not a single straight segment --
    this camera's lines visibly curve (barrel distortion), so a straight
    2-point line cuts the corner. Click along the curve, then "Finish this
    line" (min 2 points).
  - The NET is a 4-corner RECTANGLE (top-left, top-right, bottom-right,
    bottom-left of its visible face), not a single line -- the net has real
    height in the image, not just a top cable.

Nothing is derived or extrapolated between lines: a line you skip (not
visible in this camera's framing) is simply absent, never guessed.

Saves a calibration JSON with each line's points (or null if skipped),
ready for src/court_lines.py to draw directly (no detection, no derivation,
at video-processing time).
"""

import argparse
import json
import os

import cv2
from PIL import Image, ImageDraw, ImageTk
import tkinter as tk
from tkinter import messagebox

# (key, title, detail, kind, generic schematic highlight)
# kind: "polyline" (N points, user finishes manually) or "polygon" (fixed 4 points)
FEATURES = [
    ("net", "NET", "Click around the net's visible face: top-left, top-right, bottom-right, bottom-left",
     "polygon", "edge_net"),
    ("baseline", "NEAR BASELINE", "Click along the curve from left to right, then Finish",
     "polyline", "edge_baseline"),
    ("left_sideline", "NEAR LEFT SIDELINE", "Click along the curve from the net down to the near baseline, then Finish",
     "polyline", "edge_left"),
    ("right_sideline", "NEAR RIGHT SIDELINE", "Click along the curve from the net down to the near baseline, then Finish",
     "polyline", "edge_right"),
    ("kitchen_line", "NEAR KITCHEN LINE", "Click along the curve from left to right, then Finish",
     "polyline", "edge_kitchen"),
    ("center_line", "NEAR CENTER LINE", "Click along the curve from the kitchen line to the baseline, then Finish",
     "polyline", "edge_center"),
    ("far_kitchen_line", "FAR KITCHEN LINE", "The non-volley line on the OTHER side of the net. Click left to right, then Finish",
     "polyline", "far_edge_kitchen"),
    ("far_left_sideline", "FAR LEFT SIDELINE", "From the net up to the far baseline, left edge. Click, then Finish",
     "polyline", "far_edge_left"),
    ("far_right_sideline", "FAR RIGHT SIDELINE", "From the net up to the far baseline, right edge. Click, then Finish",
     "polyline", "far_edge_right"),
    ("far_center_line", "FAR CENTER LINE", "Divides the far service boxes, kitchen line to far baseline",
     "polyline", "far_edge_center"),
    ("far_baseline", "FAR BASELINE", "The far boundary line, if visible past the net. Click, then Finish",
     "polyline", "far_edge_baseline"),
]

NET_CORNER_LABELS = ["top-left", "top-right", "bottom-right", "bottom-left"]

# Rough expected pixel location of each feature's reference points in
# picklecam.mp4's frame, purely to center the zoomed-in hint crop -- never
# used in the saved calibration itself.
EXPECTED_HINT_LOCATION = {
    "net": [(463, 295), (1430, 295), (1430, 355), (463, 355)],
    "baseline": [(84, 900), (780, 1010), (1448, 900)],
    "left_sideline": [(463, 295), (270, 600), (84, 900)],
    "right_sideline": [(1430, 295), (1440, 600), (1448, 900)],
    "kitchen_line": [(530, 465), (1000, 465), (1467, 465)],
    "center_line": [(933, 465), (930, 740), (933, 1016)],
    "far_kitchen_line": [(620, 275), (950, 268), (1290, 275)],
    "far_left_sideline": [(480, 275), (560, 230), (620, 190)],
    "far_right_sideline": [(1410, 275), (1340, 230), (1280, 190)],
    "far_center_line": [(950, 270), (950, 220), (950, 170)],
    "far_baseline": [(620, 190), (950, 170), (1280, 190)],
}

HINT_CROP_SIZE = 260
SCHEMATIC_SIZE = 260

MAX_DISPLAY_W = 1920  # hard cap regardless of screen size ("max 1080p")
MAX_DISPLAY_H = 1080
SIDE_PANEL_RESERVE_W = 320
CHROME_RESERVE_H = 220  # status label + buttons + window chrome


def _compute_display_scale(img_w: int, img_h: int, root: tk.Tk) -> float:
    """Fit the image into the available screen space (minus room for the side
    panel and buttons), capped at 1920x1080, without ever upscaling -- so the
    same window works whether the source is 1080p, 4K, or a phone-portrait
    clip, and on any size monitor."""
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    avail_w = min(MAX_DISPLAY_W, screen_w - SIDE_PANEL_RESERVE_W)
    avail_h = min(MAX_DISPLAY_H, screen_h - CHROME_RESERVE_H)
    return min(avail_w / img_w, avail_h / img_h, 1.0)


# y: 0 = far baseline, 0.5 = net, 1 = near baseline (full-court schematic)
_SCHEMATIC_EDGES = {
    "edge_net": [(0, 0.47), (1, 0.47), (1, 0.53), (0, 0.53)],
    "edge_baseline": [(0, 1), (1, 1)],
    "edge_left": [(0, 0.5), (0, 1)],
    "edge_right": [(1, 0.5), (1, 1)],
    "edge_kitchen": [(0, 0.66), (1, 0.66)],
    "edge_center": [(0.5, 0.66), (0.5, 1)],
    "far_edge_kitchen": [(0, 0.34), (1, 0.34)],
    "far_edge_left": [(0, 0), (0, 0.5)],
    "far_edge_right": [(1, 0), (1, 0.5)],
    "far_edge_center": [(0.5, 0), (0.5, 0.34)],
    "far_edge_baseline": [(0, 0), (1, 0)],
}


def _make_schematic(highlight: str | None) -> Image.Image:
    """Tiny top-down diagram of the FULL court (near + far half, net in the
    middle) with the relevant feature highlighted in red, so the terminology
    is unambiguous."""
    W, H = SCHEMATIC_SIZE, SCHEMATIC_SIZE
    img = Image.new("RGB", (W, H), "#2b2b3d")
    draw = ImageDraw.Draw(img)

    margin = 30
    x0, y0, x1, y1 = margin, margin, W - margin, H - margin
    draw.rectangle((x0, y0, x1, y1), outline="white", width=2)

    def frac_to_px(fx, fy):
        return (x0 + (x1 - x0) * fx, y0 + (y1 - y0) * fy)

    near_kitchen_y = y0 + (y1 - y0) * 0.66
    far_kitchen_y = y0 + (y1 - y0) * 0.34
    net_y = y0 + (y1 - y0) * 0.5
    mid_x = (x0 + x1) / 2

    draw.line((x0, near_kitchen_y, x1, near_kitchen_y), fill="#7fd3ff", width=2)
    draw.line((x0, far_kitchen_y, x1, far_kitchen_y), fill="#7fd3ff", width=2)
    draw.line((mid_x, near_kitchen_y, mid_x, y1), fill="white", width=1)
    draw.line((mid_x, y0, mid_x, far_kitchen_y), fill="white", width=1)
    draw.line((x0, net_y, x1, net_y), fill="#888", width=4)  # net

    if highlight in _SCHEMATIC_EDGES:
        pts = [frac_to_px(*p) for p in _SCHEMATIC_EDGES[highlight]]
        if highlight == "edge_net":
            draw.polygon(pts, outline="red", width=4)
        else:
            draw.line([c for p in pts for c in p], fill="red", width=4)

    draw.text((margin, H - 18), "NEAR BASELINE", fill="#aaa")
    draw.text((margin, 5), "FAR BASELINE", fill="#aaa")
    return img


def _make_hint_crop(base_pil: Image.Image, key: str, point_idx: int) -> Image.Image:
    hints = EXPECTED_HINT_LOCATION[key]
    cx, cy = hints[min(point_idx, len(hints) - 1)]
    half = HINT_CROP_SIZE // 2
    box = (cx - half, cy - half, cx + half, cy + half)
    crop = base_pil.crop(box)
    draw = ImageDraw.Draw(crop)
    draw.line((half - 15, half, half + 15, half), fill="red", width=2)
    draw.line((half, half - 15, half, half + 15), fill="red", width=2)
    draw.ellipse((half - 20, half - 20, half + 20, half + 20), outline="red", width=2)
    return crop


class CalibrationUI:
    def __init__(self, root, image_path: str, out_path: str):
        self.root = root
        self.root.title("Calibrate the court")
        self.out_path = out_path

        bgr = cv2.imread(image_path)
        self.img_h, self.img_w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self.base_pil = Image.fromarray(rgb)  # full resolution; saved coordinates stay in THIS space

        self.scale = _compute_display_scale(self.img_w, self.img_h, root)
        disp_w, disp_h = round(self.img_w * self.scale), round(self.img_h * self.scale)
        display_pil = self.base_pil.resize((disp_w, disp_h), Image.LANCZOS) if self.scale < 1.0 else self.base_pil
        self.tk_img = ImageTk.PhotoImage(display_pil)
        self.disp_w, self.disp_h = disp_w, disp_h

        # name -> {"type": "polyline"|"polygon", "points": [...]} | None (skipped)
        self.features: dict[str, dict | None] = {}
        self._current_points: list = []  # points clicked so far for the feature in progress
        self._side_imgs = []  # keep PhotoImage refs alive

        main = tk.Frame(root)
        main.pack(fill=tk.BOTH, expand=True)

        left = tk.Frame(main)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.status = tk.Label(left, font=("Segoe UI", 12, "bold"), fg="#1a5fb4", justify=tk.LEFT)
        self.status.pack(pady=8, anchor=tk.W, padx=8)

        self.canvas = tk.Canvas(left, width=self.disp_w, height=self.disp_h)
        self.canvas.pack()
        self.canvas_img = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)
        self.canvas.bind("<Button-1>", self.on_click)

        bar = tk.Frame(left, pady=8)
        bar.pack()
        self.finish_btn = tk.Button(bar, text="Finish this line", command=self.finish_line)
        self.finish_btn.pack(side=tk.LEFT, padx=5)
        tk.Button(bar, text="Skip this line", command=self.skip_feature).pack(side=tk.LEFT, padx=5)
        tk.Button(bar, text="Undo last point", command=self.undo).pack(side=tk.LEFT, padx=5)
        tk.Button(bar, text="Start over", command=self.reset).pack(side=tk.LEFT, padx=5)
        self.save_btn = tk.Button(bar, text="Save calibration", command=self.save, bg="#4CAF50", fg="white", state=tk.DISABLED)
        self.save_btn.pack(side=tk.LEFT, padx=5)

        # Default OFF: while tracing a new line, already-finished lines are hidden
        # so they can't be confused with (or obscure) the one you're currently
        # drawing, especially where lines overlap or cross near the net.
        self.show_finished = tk.BooleanVar(value=False)
        tk.Checkbutton(
            left, text="Show already-traced lines (for reference)",
            variable=self.show_finished, command=self._redraw,
        ).pack(pady=(0, 4))

        right = tk.Frame(main, padx=15, pady=15, bg="#1e1e2e")
        right.pack(side=tk.RIGHT, fill=tk.Y)

        tk.Label(right, text="Which part?", font=("Segoe UI", 11, "bold"), fg="white", bg="#1e1e2e").pack(pady=(0, 4))
        self.schematic_label = tk.Label(right, bg="#1e1e2e")
        self.schematic_label.pack()

        tk.Label(right, text="Roughly here in the photo:", font=("Segoe UI", 11, "bold"), fg="white", bg="#1e1e2e").pack(pady=(14, 4))
        self.hint_label = tk.Label(right, bg="#1e1e2e")
        self.hint_label.pack()

        tk.Label(
            right,
            text="(The hint crop is just an approximate guide.\n\n"
                 "Lines: click along the curve, as many points\nas you need, then \"Finish this line\".\n\n"
                 "Net: click its 4 corners in order.\n\n"
                 "Not visible in this camera's framing?\nUse \"Skip this line\".)",
            font=("Segoe UI", 9), fg="#aaa", bg="#1e1e2e", justify=tk.LEFT,
        ).pack(pady=(10, 0))

        tk.Label(right, text="Redo a specific part:", font=("Segoe UI", 10, "bold"), fg="white", bg="#1e1e2e").pack(pady=(14, 4), anchor=tk.W)
        redo_box = tk.Frame(right, bg="#1e1e2e")
        redo_box.pack(fill=tk.X)
        for key, title, *_rest in FEATURES:
            tk.Button(
                redo_box, text=title, font=("Segoe UI", 8), anchor=tk.W,
                command=lambda k=key: self.redo_part(k),
            ).pack(fill=tk.X, pady=1)

        self.reset()
        self._load_existing()

    def _load_existing(self):
        """Resumes a previously saved calibration (e.g. you traced the near
        side in an earlier session and are now adding the far side) instead
        of making you re-trace everything from scratch."""
        if not os.path.exists(self.out_path):
            return
        with open(self.out_path) as f:
            saved = json.load(f).get("lines", {})
        loaded = 0
        for key, *_ in FEATURES:
            if key not in saved:
                continue
            feat = saved[key]
            self.features[key] = None if feat is None else {
                "type": feat["type"], "points": [tuple(p) for p in feat["points"]]
            }
            loaded += 1
        if loaded:
            self._redraw()
            self._update_status()
            self.save_btn.config(state=tk.NORMAL)

    def _current_feature_index(self):
        """Index of the first part NOT yet filled, in FEATURES order -- not
        just a count. This is what makes redo_part() work: clearing one
        specific already-finished part (even out of order) correctly resumes
        right there, without needing to also clear everything traced after it."""
        for i, (key, *_rest) in enumerate(FEATURES):
            if key not in self.features:
                return i
        return len(FEATURES)

    def redo_part(self, key: str):
        """Clears just ONE specific part (whatever step it's at) and jumps
        back to re-trace it, leaving every other already-traced part alone."""
        if key in self.features:
            del self.features[key]
        self._current_points = []
        self._redraw()
        self._update_status()
        self.save_btn.config(state=tk.NORMAL if self.features else tk.DISABLED)

    def reset(self):
        self.features = {}
        self._current_points = []
        self.save_btn.config(state=tk.DISABLED)
        self._redraw()
        self._update_status()

    def undo(self):
        if self._current_points:
            self._current_points.pop()
        elif self.features:
            last_key = list(self.features)[-1]
            del self.features[last_key]
        self._redraw()
        self._update_status()
        self.save_btn.config(state=tk.NORMAL if self.features else tk.DISABLED)

    def skip_feature(self):
        idx = self._current_feature_index()
        if idx >= len(FEATURES):
            return
        key, *_ = FEATURES[idx]
        self.features[key] = None
        self._current_points = []
        self._redraw()
        self._update_status()
        self.save_btn.config(state=tk.NORMAL)

    def finish_line(self):
        idx = self._current_feature_index()
        if idx >= len(FEATURES):
            return
        key, _, _, kind, _ = FEATURES[idx]
        if kind != "polyline" or len(self._current_points) < 2:
            return
        self.features[key] = {"type": "polyline", "points": list(self._current_points)}
        self._current_points = []
        self._redraw()
        self._update_status()
        self.save_btn.config(state=tk.NORMAL)

    def _update_status(self):
        idx = self._current_feature_index()
        if idx < len(FEATURES):
            key, title, detail, kind, _ = FEATURES[idx]
            if kind == "polygon":
                step = len(self._current_points)
                label = NET_CORNER_LABELS[step] if step < 4 else "done"
                self.status.config(text=f"Part {idx + 1}/{len(FEATURES)}: {title} -- click the {label} corner\n{detail}")
                self.finish_btn.config(state=tk.DISABLED)
            else:
                n = len(self._current_points)
                self.status.config(
                    text=f"Part {idx + 1}/{len(FEATURES)}: {title} -- {n} point(s) clicked\n{detail}"
                )
                self.finish_btn.config(state=tk.NORMAL if n >= 2 else tk.DISABLED)
            self._update_side_panel(key, len(self._current_points))
        else:
            self.status.config(text="All parts done (traced or skipped). Review below, then Save.")
            self.finish_btn.config(state=tk.DISABLED)
            self.show_finished.set(True)  # nothing left in progress to be obscured -- show everything for review
            self._redraw()
            self._update_side_panel(None, 0)

    def _update_side_panel(self, key, point_idx):
        self._side_imgs.clear()

        idx = self._current_feature_index()
        highlight = FEATURES[idx][4] if idx < len(FEATURES) else None
        schematic = _make_schematic(highlight)
        schematic_tk = ImageTk.PhotoImage(schematic)
        self._side_imgs.append(schematic_tk)
        self.schematic_label.config(image=schematic_tk)

        if key is not None:
            hint = _make_hint_crop(self.base_pil, key, point_idx)
            hint_tk = ImageTk.PhotoImage(hint)
            self._side_imgs.append(hint_tk)
            self.hint_label.config(image=hint_tk)
        else:
            self.hint_label.config(image="")

    def on_click(self, event):
        idx = self._current_feature_index()
        if idx >= len(FEATURES):
            return
        key, _, _, kind, _ = FEATURES[idx]
        # Click lands in display coordinates; store in the ORIGINAL image's
        # coordinate space so calibration stays correct regardless of scale.
        pt = (event.x / self.scale, event.y / self.scale)
        self._current_points.append(pt)

        if kind == "polygon" and len(self._current_points) == 4:
            self.features[key] = {"type": "polygon", "points": list(self._current_points)}
            self._current_points = []
            self.save_btn.config(state=tk.NORMAL)

        self._redraw()
        self._update_status()

    def _redraw(self):
        self.canvas.delete("overlay")

        def to_disp(p):
            return p[0] * self.scale, p[1] * self.scale

        if self.show_finished.get():
            for key, feat in self.features.items():
                if feat is None:
                    continue
                pts = [to_disp(p) for p in feat["points"]]
                closed = feat["type"] == "polygon"
                seq = pts + [pts[0]] if closed else pts
                for i in range(len(seq) - 1):
                    self.canvas.create_line(*seq[i], *seq[i + 1], fill="#00e5ff", width=3, tags="overlay")
                mx, my = pts[len(pts) // 2]
                self.canvas.create_text(mx, my - 10, text=key, fill="#00e5ff", font=("Segoe UI", 10, "bold"), tags="overlay")

        for p in self._current_points:
            dx, dy = to_disp(p)
            self.canvas.create_oval(dx - 5, dy - 5, dx + 5, dy + 5, fill="red", outline="white", tags="overlay")
        for i in range(len(self._current_points) - 1):
            self.canvas.create_line(*to_disp(self._current_points[i]), *to_disp(self._current_points[i + 1]),
                                     fill="red", width=2, tags="overlay")

    def save(self):
        data = {
            "lines": {
                key: ({"type": feat["type"], "points": [list(p) for p in feat["points"]]} if feat is not None else None)
                for key, feat in self.features.items()
            }
        }
        os.makedirs(os.path.dirname(self.out_path) or ".", exist_ok=True)
        with open(self.out_path, "w") as f:
            json.dump(data, f, indent=2)
        n_drawn = sum(1 for v in self.features.values() if v is not None)
        n_skipped = sum(1 for v in self.features.values() if v is None)
        messagebox.showinfo("Saved", f"Calibration saved to {self.out_path}\n({n_drawn} part(s) traced, {n_skipped} skipped)")
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="output/court_background.jpg")
    parser.add_argument("--out", default="output/calibration/picklecam_vertical.json")
    args = parser.parse_args()

    root = tk.Tk()
    CalibrationUI(root, args.image, args.out)
    root.mainloop()


if __name__ == "__main__":
    main()
