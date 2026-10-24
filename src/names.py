import random

_NAME_POOL = [
    "Ace", "Blaze", "Comet", "Dash", "Echo", "Falcon", "Gizmo", "Hawk",
    "Indigo", "Jester", "Kiwi", "Lynx", "Mango", "Nova", "Onyx", "Pixel",
    "Quartz", "Rocket", "Sable", "Tango", "Umber", "Viper", "Willow", "Xena",
    "Yonder", "Zephyr", "Bramble", "Cinder", "Drift", "Ember",
]


class NameAssigner:
    """Assigns a random, stable name to each track_id the first time it's seen."""

    def __init__(self, seed: int | None = None):
        self._pool = list(_NAME_POOL)
        random.Random(seed).shuffle(self._pool)
        self._assigned: dict[int, str] = {}

    def get(self, track_id: int) -> str:
        if track_id not in self._assigned:
            self._assigned[track_id] = self.new()
        return self._assigned[track_id]

    def new(self) -> str:
        if self._pool:
            return self._pool.pop()
        self._counter = getattr(self, "_counter", 0) + 1
        return f"Player{self._counter}"
