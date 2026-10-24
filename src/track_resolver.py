"""Delays naming a brand-new tracker ID for up to `confirm_seconds`, retrying
the gallery match on every individual frame (not a blended average -- that
washes out the exact signal that makes a match succeed) until one scores
high enough. If nothing matches by the deadline, the whole buffered burst of
frames becomes a fresh identity's starting set of reference exemplars.
"""

import numpy as np

from src.reid_gallery import ReIDGallery


class TrackResolver:
    def __init__(self, gallery: ReIDGallery, name_assigner, fps: float = 30.0, confirm_seconds: float = 1.0):
        self.gallery = gallery
        self.name_assigner = name_assigner
        self.confirm_frames = max(1, round(fps * confirm_seconds))

        self._resolved: dict[int, str] = {}
        self._pending_hists: dict[int, list[np.ndarray]] = {}
        self._pending_since: dict[int, int] = {}

    def resolve(self, tid: int, frame, box_xyxy, frame_idx: int) -> str:
        if tid in self._resolved:
            name = self._resolved[tid]
            self.gallery.mark_active(name, frame_idx)
            return name

        hist = self.gallery._histogram(self.gallery.crop_box(frame, box_xyxy))
        if hist is None:
            # No usable crop (off-screen edge, etc.) -- don't let this stall the track forever.
            name = self.name_assigner.new()
            self.gallery.register_new(name, np.zeros(960, dtype="float32"), frame_idx)
            self._resolved[tid] = name
            return name

        self._pending_hists.setdefault(tid, []).append(hist)
        self._pending_since.setdefault(tid, frame_idx)

        # Try THIS frame's own histogram against the gallery -- not an average of
        # everything buffered so far, which would blur out the match signal.
        best_name, best_score = self.gallery.best_match(hist, frame_idx)
        elapsed = frame_idx - self._pending_since[tid]

        if best_name is not None and best_score >= self.gallery.match_thresh:
            self.gallery.register_match(best_name, hist, frame_idx)
            self._resolved[tid] = best_name
            del self._pending_hists[tid]
            return best_name

        if elapsed >= self.confirm_frames:
            # No match after the whole window -- mint a new identity, seeding it
            # with every frame we buffered so it starts with several exemplars.
            name = self.name_assigner.new()
            buffered = self._pending_hists.pop(tid)
            self.gallery.register_new(name, buffered[0], frame_idx)
            for extra in buffered[1:]:
                self.gallery.register_match(name, extra, frame_idx)
            self._resolved[tid] = name
            return name

        return "..."  # still confirming -- shown briefly while we wait for a stronger signal

    def resolved_names(self) -> set[str]:
        return set(self._resolved.values())
