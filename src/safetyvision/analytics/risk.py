"""Real-time risk engine: turns tracked detections into safety events and zone risk scores."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from ..config import RiskConfig
from ..simulation.factory import FactoryLayout

FEATURES = [
    "n_person", "n_forklift", "n_no_helmet", "n_no_vest", "n_danger",
    "n_danger_no_ppe", "n_restricted", "n_near_miss", "severity",
]


class RiskEngine:
    """Rule-based risk scoring on top of perception output.

    Per frame and per zone it computes a severity score

        s_z = sum_people [ w_h*no_helmet + w_v*no_vest
                           + w_d*in_hazard_band*(1 + missing_ppe)
                           + w_r*in_restricted
                           + w_n*near_miss*(1 + no_vest) ]

    and keeps an exponentially-weighted moving average ``zone_risk`` used for
    live dashboards and as the hazard prior for the MARL inspectors.
    """

    def __init__(self, layout: FactoryLayout, cfg: RiskConfig):
        self.layout = layout
        self.cfg = cfg
        nz = layout.cfg.n_zones
        self.zone_risk = np.zeros(nz)
        self.cum_risk = np.zeros(nz)
        self.events: list[dict] = []
        self.frame_features: list[np.ndarray] = []
        self.safety_index: list[float] = []
        self._streak: dict[tuple[int, str], int] = defaultdict(int)

    def update(self, frame: int, tracks) -> np.ndarray:
        cfg, L = self.cfg, self.layout
        feats = np.zeros((L.cfg.n_zones, len(FEATURES)))
        people = [t for t in tracks if t.cls == "person"]
        forks = [t for t in tracks if t.cls == "forklift"]
        active: set[tuple[int, str]] = set()
        for f in forks:
            feats[L.zone_of(f.y, f.x), 1] += 1
        for p in people:
            z = L.zone_of(p.y, p.x)
            no_h, no_v = (not p.helmet), (not p.vest)
            danger = L.in_danger(p.y, p.x)
            restricted = L.in_restricted(p.y, p.x)
            near = any(np.hypot(p.y - f.y, p.x - f.x) < cfg.near_miss_px for f in forks)
            sev = (cfg.w_no_helmet * no_h + cfg.w_no_vest * no_v
                   + cfg.w_danger_zone * danger * (1 + (no_h or no_v))
                   + cfg.w_restricted * restricted
                   + cfg.w_near_miss * near * (1 + no_v))
            feats[z] += [1, 0, no_h, no_v, danger, danger and (no_h or no_v), restricted, near, sev]
            for kind, on in (("no_helmet", no_h), ("no_vest", no_v), ("hazard_zone_no_ppe", danger and (no_h or no_v)),
                             ("restricted_intrusion", restricted), ("near_miss_vehicle", near)):
                if on:
                    active.add((p.id, kind))
                    self._streak[(p.id, kind)] += 1
                    if self._streak[(p.id, kind)] == cfg.event_min_frames:
                        self.events.append(dict(frame=frame, zone=z, kind=kind, track_id=p.id,
                                                y=round(p.y, 1), x=round(p.x, 1), severity=round(sev, 2)))
        for key in list(self._streak):
            if key not in active:
                del self._streak[key]
        sev = feats[:, -1]
        a = cfg.ema_alpha
        self.zone_risk = (1 - a) * self.zone_risk + a * sev
        self.cum_risk += sev
        self.frame_features.append(feats)
        self.safety_index.append(float(100.0 * np.exp(-self.zone_risk.sum() / 4.0)))
        return self.zone_risk

    # ------------------------------------------------------------------ reporting helpers
    def events_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.events, columns=["frame", "zone", "kind", "track_id", "y", "x", "severity"])

    def feature_tensor(self) -> np.ndarray:
        """(frames, zones, features)"""
        return np.stack(self.frame_features)

    def risk_grid(self, values: np.ndarray | None = None) -> np.ndarray:
        c = self.layout.cfg
        v = self.cum_risk if values is None else values
        return v.reshape(c.n_rows, c.n_cols)
