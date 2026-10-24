import json

import cv2
import numpy as np


class ReIDGallery:
    """Persistent appearance-based re-identification using HSV color histograms.

    The built-in tracker only remembers a lost track for a short buffer window
    (~5s). This gallery remembers every player's appearance for the whole
    video, so when a brand-new tracker ID shows up (someone who left the
    frame and came back, or whose ID switched), we check whether it matches
    someone we've already named before handing out a new name.

    Each name keeps a small LIST of reference histograms (exemplars) rather
    than one blended average -- the same shirt looks different under
    different lighting/pose, and averaging those into one "mushy" centroid
    washes out the signal that would otherwise match at least one of them.
    A candidate is matched against the single best-scoring exemplar.
    """

    def __init__(self, match_thresh: float = 0.75, active_window: int = 5, max_exemplars: int = 25):
        self.match_thresh = match_thresh
        self.active_window = active_window
        self.max_exemplars = max_exemplars
        self._gallery: dict[str, list[np.ndarray]] = {}
        self._last_seen_frame: dict[str, int] = {}

    @staticmethod
    def _torso_region(crop):
        """Middle-of-body slice (skips head/cap and shoes, trims background at the edges)."""
        h, w = crop.shape[:2]
        y0, y1 = int(h * 0.2), int(h * 0.8)
        x0, x1 = int(w * 0.15), int(w * 0.85)
        region = crop[y0:y1, x0:x1]
        return region if region.size else crop

    @staticmethod
    def _histogram(crop) -> np.ndarray | None:
        if crop.size == 0:
            return None
        crop = ReIDGallery._torso_region(crop)
        if crop.size == 0:
            return None
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return hist.flatten().astype("float32")

    def mark_active(self, name: str, frame_idx: int) -> None:
        self._last_seen_frame[name] = frame_idx

    @staticmethod
    def crop_box(frame, box_xyxy):
        x1, y1, x2, y2 = [int(v) for v in box_xyxy]
        x1, y1 = max(x1, 0), max(y1, 0)
        return frame[y1:y2, x1:x2]

    @staticmethod
    def _best_score_against(hist: np.ndarray, exemplars: list[np.ndarray]) -> float:
        return max(cv2.compareHist(hist, ex, cv2.HISTCMP_CORREL) for ex in exemplars)

    def best_match(self, hist: np.ndarray, frame_idx: int) -> tuple[str | None, float]:
        """Best-scoring gallery name for `hist` (best exemplar per name), excluding
        names seen very recently -- someone already active on screen under another
        track can't be this detection."""
        best_name, best_score = None, -1.0
        for name, exemplars in self._gallery.items():
            if frame_idx - self._last_seen_frame.get(name, -10**9) <= self.active_window:
                continue
            score = self._best_score_against(hist, exemplars)
            if score > best_score:
                best_score, best_name = score, name
        return best_name, best_score

    def register_match(self, name: str, hist: np.ndarray, frame_idx: int) -> None:
        exemplars = self._gallery[name]
        exemplars.append(hist)
        if len(exemplars) > self.max_exemplars:
            del exemplars[0]
        self._last_seen_frame[name] = frame_idx

    def register_new(self, name: str, hist: np.ndarray, frame_idx: int) -> None:
        self._gallery[name] = [hist]
        self._last_seen_frame[name] = frame_idx

    def identify(self, frame, box_xyxy, frame_idx: int, new_name_fn) -> str:
        """One-shot match (single frame, decide immediately). See TrackResolver for
        the delayed/averaged version used by the live tracking run."""
        hist = self._histogram(self.crop_box(frame, box_xyxy))
        if hist is None:
            name = new_name_fn()
            self._last_seen_frame[name] = frame_idx
            return name

        best_name, best_score = self.best_match(hist, frame_idx)
        if best_name is not None and best_score >= self.match_thresh:
            self.register_match(best_name, hist, frame_idx)
            return best_name

        name = new_name_fn()
        self.register_new(name, hist, frame_idx)
        return name

    def __len__(self) -> int:
        return len(self._gallery)

    def rename(self, old: str, new: str) -> None:
        if old == new or old not in self._gallery:
            return
        self._gallery[new] = self._gallery.pop(old)
        if old in self._last_seen_frame:
            self._last_seen_frame[new] = self._last_seen_frame.pop(old)

    def save(self, path: str) -> None:
        data = {name: [h.tolist() for h in exemplars] for name, exemplars in self._gallery.items()}
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path: str) -> None:
        with open(path) as f:
            data = json.load(f)
        for name, value in data.items():
            # Accept both the new (list-of-exemplars) and old (single averaged
            # histogram) gallery.json formats.
            exemplars = value if (value and isinstance(value[0], list)) else [value]
            self._gallery[name] = [np.array(h, dtype="float32") for h in exemplars]
            self._last_seen_frame[name] = -(10**9)
