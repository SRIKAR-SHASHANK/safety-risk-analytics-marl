"""Agent-based simulator of a manufacturing shop floor.

The simulator produces *ground truth* (worker positions, PPE state, forklift
positions and stochastic safety incidents). The computer-vision stack never
sees this ground truth directly: it only receives rendered, degraded camera
frames (see :mod:`safetyvision.vision.renderer`). Ground truth is used only to
train/evaluate the perception models and to label incidents for the
predictive-analytics stage.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import PlantConfig


@dataclass(frozen=True)
class Rect:
    y0: int
    x0: int
    y1: int
    x1: int

    def contains(self, y: float, x: float) -> bool:
        return self.y0 <= y < self.y1 and self.x0 <= x < self.x1

    def expand(self, m: int) -> "Rect":
        return Rect(self.y0 - m, self.x0 - m, self.y1 + m, self.x1 + m)

    @property
    def center(self) -> tuple[float, float]:
        return (self.y0 + self.y1) / 2.0, (self.x0 + self.x1) / 2.0


@dataclass
class Worker:
    id: int
    y: float
    x: float
    ty: float
    tx: float
    speed: float
    helmet: bool
    vest: bool
    propensity: float          # individual risk-taking tendency (>1 = riskier)
    dwell: int = 0


@dataclass
class Forklift:
    id: int
    s: float                   # arc-length position along the aisle loop
    speed: float
    y: float = 0.0
    x: float = 0.0
    horizontal: bool = True


@dataclass
class Incident:
    frame: int
    zone: int
    kind: str
    worker_id: int


@dataclass
class FrameState:
    frame: int
    workers: list = field(default_factory=list)     # list[dict]
    forklifts: list = field(default_factory=list)   # list[dict]
    incidents: list = field(default_factory=list)   # list[Incident]


class FactoryLayout:
    """Static plant layout: machines, hazard bands, restricted area, aisles."""

    def __init__(self, cfg: PlantConfig):
        self.cfg = cfg
        self.machines = [
            Rect(18, 20, 58, 72), Rect(20, 110, 56, 150), Rect(18, 196, 60, 262),
            Rect(152, 28, 202, 82), Rect(160, 190, 212, 242), Rect(152, 268, 196, 304),
        ]
        self.danger = [m.expand(cfg.danger_margin) for m in self.machines]
        self.restricted = [Rect(166, 106, 226, 164)]          # chemical / battery store
        # forklift loop (y, x) corners -> two one-way lanes along the main aisle
        self.loop = np.array([(104, 14), (104, 306), (136, 306), (136, 14)], dtype=float)
        seg = np.roll(self.loop, -1, axis=0) - self.loop
        self.seg_len = np.abs(seg).sum(axis=1)
        self.perimeter = float(self.seg_len.sum())

    # ------------------------------------------------------------------ helpers
    def zone_of(self, y: float, x: float) -> int:
        zs = self.cfg.zone_size
        r = int(np.clip(y // zs, 0, self.cfg.n_rows - 1))
        c = int(np.clip(x // zs, 0, self.cfg.n_cols - 1))
        return r * self.cfg.n_cols + c

    def in_machine(self, y: float, x: float) -> bool:
        return any(m.contains(y, x) for m in self.machines)

    def in_danger(self, y: float, x: float) -> bool:
        return any(d.contains(y, x) for d in self.danger) and not self.in_machine(y, x)

    def in_restricted(self, y: float, x: float) -> bool:
        return any(r.contains(y, x) for r in self.restricted)

    def loop_point(self, s: float) -> tuple[float, float, bool]:
        s = s % self.perimeter
        for i, L in enumerate(self.seg_len):
            if s <= L:
                a, b = self.loop[i], self.loop[(i + 1) % len(self.loop)]
                p = a + (b - a) * (s / L)
                return float(p[0]), float(p[1]), bool(a[0] == b[0])
            s -= L
        a = self.loop[0]
        return float(a[0]), float(a[1]), True

    def zone_static_features(self) -> np.ndarray:
        """Per-zone static descriptors: [machine-hazard fraction, restricted fraction, aisle fraction]."""
        cfg = self.cfg
        H, W = cfg.height, cfg.width
        yy, xx = np.mgrid[0:H, 0:W]
        danger = np.zeros((H, W), bool)
        for d in self.danger:
            danger |= (yy >= d.y0) & (yy < d.y1) & (xx >= d.x0) & (xx < d.x1)
        restricted = np.zeros((H, W), bool)
        for r in self.restricted:
            restricted |= (yy >= r.y0) & (yy < r.y1) & (xx >= r.x0) & (xx < r.x1)
        aisle = (yy >= 96) & (yy < 144)
        zs = cfg.zone_size
        feats = np.zeros((cfg.n_zones, 3))
        for z in range(cfg.n_zones):
            r, c = divmod(z, cfg.n_cols)
            sl = (slice(r * zs, (r + 1) * zs), slice(c * zs, (c + 1) * zs))
            feats[z] = [danger[sl].mean(), restricted[sl].mean(), aisle[sl].mean()]
        return feats


class FactorySimulator:
    """Steps workers and forklifts and samples ground-truth incidents."""

    def __init__(self, cfg: PlantConfig, seed: int = 0):
        self.cfg = cfg
        self.layout = FactoryLayout(cfg)
        self.rng = np.random.default_rng(seed)
        self.frame = 0
        self.workers: list[Worker] = []
        self.forklifts: list[Forklift] = []
        self._init_agents()

    # ------------------------------------------------------------------ setup
    def _free_point(self) -> tuple[float, float]:
        H, W = self.cfg.height, self.cfg.width
        for _ in range(200):
            y, x = self.rng.uniform(6, H - 6), self.rng.uniform(6, W - 6)
            if not self.layout.in_machine(y, x):
                return y, x
        return H / 2, W / 2

    def _workstation_point(self) -> tuple[float, float]:
        """A point inside a machine's hazard band (where operators stand)."""
        m = self.layout.machines[self.rng.integers(len(self.layout.machines))]
        d = self.cfg.danger_margin
        for _ in range(100):
            y = self.rng.uniform(m.y0 - d + 2, m.y1 + d - 2)
            x = self.rng.uniform(m.x0 - d + 2, m.x1 + d - 2)
            if self.layout.in_danger(y, x):
                return y, x
        return self._free_point()

    def _restricted_point(self) -> tuple[float, float]:
        r = self.layout.restricted[0]
        return self.rng.uniform(r.y0 + 4, r.y1 - 4), self.rng.uniform(r.x0 + 4, r.x1 - 4)

    def _pick_target(self, w: Worker) -> tuple[float, float]:
        u = self.rng.random()
        p_restricted = 0.04 * w.propensity
        if u < p_restricted:
            return self._restricted_point()
        if u < p_restricted + 0.55:
            return self._workstation_point()
        return self._free_point()

    def _init_agents(self) -> None:
        cfg = self.cfg
        for i in range(cfg.n_workers):
            y, x = self._free_point()
            w = Worker(
                id=i, y=y, x=x, ty=y, tx=x,
                speed=float(self.rng.uniform(1.2, 2.4)),
                helmet=bool(self.rng.random() > cfg.p_no_helmet),
                vest=bool(self.rng.random() > cfg.p_no_vest),
                propensity=float(self.rng.lognormal(0.0, 0.5)),
            )
            w.ty, w.tx = self._pick_target(w)
            self.workers.append(w)
        for j in range(cfg.n_forklifts):
            f = Forklift(id=j, s=j * self.layout.perimeter / cfg.n_forklifts,
                         speed=float(self.rng.uniform(2.5, 3.5)))
            f.y, f.x, f.horizontal = self.layout.loop_point(f.s)
            self.forklifts.append(f)

    # ------------------------------------------------------------------ dynamics
    def _move_worker(self, w: Worker) -> None:
        if w.dwell > 0:
            w.dwell -= 1
            # small fidgeting while working
            w.y += self.rng.normal(0, 0.3)
            w.x += self.rng.normal(0, 0.3)
        else:
            dy, dx = w.ty - w.y, w.tx - w.x
            dist = np.hypot(dy, dx)
            if dist < w.speed:
                w.y, w.x = w.ty, w.tx
                w.dwell = int(self.rng.integers(10, 60))
                w.ty, w.tx = self._pick_target(w)
            else:
                ny, nx = w.y + w.speed * dy / dist, w.x + w.speed * dx / dist
                if self.layout.in_machine(ny, nx):   # walk around machines
                    ny, nx = w.y + self.rng.normal(0, 2), w.x + self.rng.normal(0, 2)
                w.y, w.x = ny, nx
        w.y = float(np.clip(w.y, 5, self.cfg.height - 6))
        w.x = float(np.clip(w.x, 5, self.cfg.width - 6))
        # PPE compliance dynamics
        cfg = self.cfg
        if w.helmet and self.rng.random() < cfg.ppe_drop_rate * w.propensity:
            w.helmet = False
        elif not w.helmet and self.rng.random() < cfg.ppe_fix_rate:
            w.helmet = True
        if w.vest and self.rng.random() < cfg.ppe_drop_rate * w.propensity:
            w.vest = False
        elif not w.vest and self.rng.random() < cfg.ppe_fix_rate:
            w.vest = True

    def _sample_incidents(self) -> list[Incident]:
        out: list[Incident] = []
        L = self.layout
        for w in self.workers:
            z = L.zone_of(w.y, w.x)
            for f in self.forklifts:
                d = np.hypot(w.y - f.y, w.x - f.x)
                if d < 10.0:
                    p = 0.03 * (1.0 if w.vest else 2.5)
                    if self.rng.random() < p:
                        out.append(Incident(self.frame, z, "struck_by_vehicle", w.id))
            if L.in_danger(w.y, w.x):
                p = 0.002 * (1.0 + 2.0 * (not w.helmet)) * w.propensity
                if self.rng.random() < p:
                    out.append(Incident(self.frame, z, "machine_contact", w.id))
            if L.in_restricted(w.y, w.x):
                if self.rng.random() < 0.004 * (1.0 + (not w.helmet)):
                    out.append(Incident(self.frame, z, "hazmat_exposure", w.id))
        return out

    def step(self) -> FrameState:
        for w in self.workers:
            self._move_worker(w)
        for f in self.forklifts:
            f.s = (f.s + f.speed) % self.layout.perimeter
            f.y, f.x, f.horizontal = self.layout.loop_point(f.s)
        incidents = self._sample_incidents()
        state = FrameState(
            frame=self.frame,
            workers=[dict(id=w.id, y=w.y, x=w.x, helmet=w.helmet, vest=w.vest) for w in self.workers],
            forklifts=[dict(id=f.id, y=f.y, x=f.x, horizontal=f.horizontal) for f in self.forklifts],
            incidents=incidents,
        )
        self.frame += 1
        return state
