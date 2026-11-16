"""Multi-object centroid tracker (Hungarian assignment) with temporal PPE voting."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..config import VisionConfig


@dataclass
class Track:
    id: int
    cls: str
    y: float
    x: float
    hits: int = 1
    missed: int = 0
    helmet_votes: deque = field(default_factory=deque)
    vest_votes: deque = field(default_factory=deque)

    @property
    def helmet(self) -> bool:
        return bool(np.mean(self.helmet_votes) >= 0.5) if self.helmet_votes else True

    @property
    def vest(self) -> bool:
        return bool(np.mean(self.vest_votes) >= 0.5) if self.vest_votes else True


class CentroidTracker:
    def __init__(self, cfg: VisionConfig):
        self.cfg = cfg
        self.tracks: dict[int, Track] = {}
        self._next = 0

    def update(self, dets, ppe: list[tuple[bool, bool]] | None = None) -> list[Track]:
        """``dets``: list of Detection; ``ppe``: per-detection (helmet, vest) predictions (persons only)."""
        cfg = self.cfg
        ppe = ppe or [(True, True)] * len(dets)
        tids = list(self.tracks)
        assigned_t, assigned_d = set(), set()
        if tids and dets:
            D = np.full((len(tids), len(dets)), 1e6)
            for a, t in enumerate(tids):
                tr = self.tracks[t]
                for b, d in enumerate(dets):
                    if d.cls == tr.cls:
                        D[a, b] = np.hypot(tr.y - d.y, tr.x - d.x)
            rows, cols = linear_sum_assignment(D)
            for a, b in zip(rows, cols):
                if D[a, b] <= cfg.track_max_dist:
                    tr = self.tracks[tids[a]]
                    tr.y, tr.x = dets[b].y, dets[b].x
                    tr.hits += 1
                    tr.missed = 0
                    self._vote(tr, ppe[b])
                    assigned_t.add(tids[a])
                    assigned_d.add(b)
        for t in tids:
            if t not in assigned_t:
                tr = self.tracks[t]
                tr.missed += 1
                if tr.missed > cfg.track_max_missed:
                    del self.tracks[t]
        for b, d in enumerate(dets):
            if b not in assigned_d:
                tr = Track(self._next, d.cls, d.y, d.x)
                self._vote(tr, ppe[b])
                self.tracks[self._next] = tr
                self._next += 1
        return [t for t in self.tracks.values() if t.missed == 0 and t.hits >= 2]

    def _vote(self, tr: Track, p: tuple[bool, bool]) -> None:
        w = self.cfg.ppe_vote_window
        tr.helmet_votes.append(bool(p[0]))
        tr.vest_votes.append(bool(p[1]))
        while len(tr.helmet_votes) > w:
            tr.helmet_votes.popleft()
            tr.vest_votes.popleft()
