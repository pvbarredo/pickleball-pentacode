"""Applies the name_mapping.json produced by label_ui.py to the raw
gallery.json from src.enroll, merging any placeholders you grouped together
under one name, and writes the final seed gallery for main.py.

Keeps every kept exemplar histogram (not a single blended average) so the
live run can match a new appearance against whichever reference looks most
like it -- averaging into one histogram washes out the signal that varying
lighting/pose would otherwise still match.
"""

import argparse
import json
import os

import cv2
import numpy as np


def _drop_outliers(hists: list, outlier_thresh: float = 0.5) -> tuple:
    """Drop any histogram that looks nothing like the group's median -- a single bad
    crop (overexposed, mostly background) shouldn't poison the reference set."""
    if len(hists) <= 2:
        return hists, 0
    stack = np.stack(hists).astype("float32")
    median = np.median(stack, axis=0).astype("float32")
    kept = [h for h in hists if cv2.compareHist(h.astype("float32"), median, cv2.HISTCMP_CORREL) >= outlier_thresh]
    if not kept:
        kept = hists
    return kept, len(hists) - len(kept)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--enroll-dir", default="output/enroll")
    parser.add_argument("--out", default="output/enroll/gallery_named.json")
    args = parser.parse_args()

    with open(os.path.join(args.enroll_dir, "gallery.json")) as f:
        raw_gallery = json.load(f)
    with open(os.path.join(args.enroll_dir, "name_mapping.json")) as f:
        mapping = json.load(f)

    # Group case/whitespace-insensitively so "Bading 2" and "bading 2" merge into one
    # person instead of splitting into two; the first-seen spelling becomes the display name.
    display_names: dict[str, str] = {}
    merged: dict[str, list[np.ndarray]] = {}
    for placeholder, final_name in mapping.items():
        value = raw_gallery.get(placeholder)
        if value is None:
            continue
        # raw gallery.json entries are lists of exemplar histograms (one list per
        # placeholder); flatten every exemplar from every placeholder sharing a name.
        exemplars = value if (value and isinstance(value[0], list)) else [value]

        key = final_name.strip().lower()
        display_names.setdefault(key, final_name.strip())
        bucket = merged.setdefault(key, [])
        bucket.extend(np.array(h, dtype="float32") for h in exemplars)

    merged = {display_names[key]: hists for key, hists in merged.items()}

    named_gallery = {}
    dropped_counts = {}
    for name, hists in merged.items():
        kept, n_dropped = _drop_outliers(hists)
        named_gallery[name] = [h.tolist() for h in kept]
        dropped_counts[name] = n_dropped

    with open(args.out, "w") as f:
        json.dump(named_gallery, f, indent=2)

    print(f"Saved {len(named_gallery)} named player(s) to {args.out}:")
    for name in named_gallery:
        n = dropped_counts[name]
        extra = f" ({n} outlier crop(s) excluded)" if n else ""
        print(f"  {name}: {len(named_gallery[name])} reference exemplar(s) from {len(merged[name])} clip(s){extra}")


if __name__ == "__main__":
    main()
