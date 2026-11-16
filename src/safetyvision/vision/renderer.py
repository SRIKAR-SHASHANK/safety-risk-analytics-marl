"""Renders the simulator state as a degraded top-down CCTV frame (RGB uint8)."""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from ..config import PlantConfig, RenderConfig
from ..simulation.factory import FactoryLayout, FrameState

# Canonical colours (RGB). The vision stack only knows these as *colour priors*
# for segmentation, exactly as a classical colour-based PPE detector would.
FLOOR = (200, 195, 180)
LANE = (225, 210, 110)
MACHINE = (70, 72, 78)
HAZARD = (185, 70, 60)
RESTRICTED = (205, 182, 205)
FORKLIFT = (0, 150, 150)
VEST = (255, 140, 0)          # hi-vis orange
SHIRT = (40, 50, 115)         # navy work shirt (no vest)
HELMET = (250, 230, 40)       # yellow hard hat
HAIR = (90, 60, 40)           # bare head

PERSON_RADIUS = 5
HEAD_RADIUS = 2


def _disk(r: int) -> np.ndarray:
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    return (yy ** 2 + xx ** 2) <= r * r + 0.5


_TORSO = _disk(PERSON_RADIUS)
_HEAD = _disk(HEAD_RADIUS)


def _stamp(img: np.ndarray, mask: np.ndarray, cy: int, cx: int, color) -> None:
    r = mask.shape[0] // 2
    H, W = img.shape[:2]
    y0, x0 = cy - r, cx - r
    ys, xs = max(0, y0), max(0, x0)
    ye, xe = min(H, y0 + mask.shape[0]), min(W, x0 + mask.shape[1])
    if ys >= ye or xs >= xe:
        return
    m = mask[ys - y0:ye - y0, xs - x0:xe - x0]
    img[ys:ye, xs:xe][m] = color


class Renderer:
    def __init__(self, layout: FactoryLayout, render_cfg: RenderConfig, seed: int = 0):
        self.layout = layout
        self.cfg = render_cfg
        self.rng = np.random.default_rng(seed)
        self.background = self._build_background(layout.cfg)

    @staticmethod
    def _build_background(cfg: PlantConfig) -> np.ndarray:
        H, W = cfg.height, cfg.width
        layout = FactoryLayout(cfg)
        img = np.empty((H, W, 3), np.uint8)
        img[:] = FLOOR
        yy, xx = np.mgrid[0:H, 0:W]
        # forklift lane markings
        for y in (96, 120, 143):
            img[y:y + 2, 8:W - 8] = LANE
        for r in layout.restricted:
            img[r.y0:r.y1, r.x0:r.x1] = RESTRICTED
        stripes = ((yy + xx) // 4) % 2 == 0
        for d, m in zip(layout.danger, layout.machines):
            band = (yy >= d.y0) & (yy < d.y1) & (xx >= d.x0) & (xx < d.x1) & stripes
            img[band] = HAZARD
            img[m.y0:m.y1, m.x0:m.x1] = MACHINE
        return img

    def render_clean(self, state: FrameState) -> np.ndarray:
        img = self.background.copy()
        for f in state.forklifts:
            cy, cx = int(round(f["y"])), int(round(f["x"]))
            hh, hw = (5, 8) if f["horizontal"] else (8, 5)
            img[max(0, cy - hh):cy + hh, max(0, cx - hw):cx + hw] = FORKLIFT
        for w in state.workers:
            cy, cx = int(round(w["y"])), int(round(w["x"]))
            _stamp(img, _TORSO, cy, cx, VEST if w["vest"] else SHIRT)
            _stamp(img, _HEAD, cy, cx, HELMET if w["helmet"] else HAIR)
        return img

    def degrade(self, img: np.ndarray) -> np.ndarray:
        cfg, rng = self.cfg, self.rng
        out = img.astype(np.float32)
        if rng.random() < cfg.occlusion_prob:
            H, W = out.shape[:2]
            for _ in range(int(rng.integers(1, 3))):
                h, w = int(rng.integers(10, 26)), int(rng.integers(10, 26))
                y, x = int(rng.integers(0, H - h)), int(rng.integers(0, W - w))
                out[y:y + h, x:x + w] = 0.5 * out[y:y + h, x:x + w] + 0.5 * 150.0  # steam / haze
        out *= rng.uniform(*cfg.illumination_range)
        if rng.random() < cfg.blur_prob:
            out = ndimage.gaussian_filter(out, sigma=(cfg.blur_sigma, cfg.blur_sigma, 0))
        out += rng.standard_normal(out.shape, dtype=np.float32) * cfg.noise_std
        return np.clip(out, 0, 255).astype(np.uint8)

    def render(self, state: FrameState) -> np.ndarray:
        return self.degrade(self.render_clean(state))
