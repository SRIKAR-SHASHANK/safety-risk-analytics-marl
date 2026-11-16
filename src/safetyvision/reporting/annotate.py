"""Draws detections, PPE status and the live zone-risk overlay on a frame (pure NumPy)."""
from __future__ import annotations

import numpy as np

GREEN = np.array([30, 200, 60], np.uint8)
RED = np.array([230, 20, 20], np.uint8)
CYAN = np.array([0, 220, 255], np.uint8)


def _box(img, y0, x0, y1, x1, color):
    H, W = img.shape[:2]
    y0, x0, y1, x1 = max(0, y0), max(0, x0), min(H - 1, y1), min(W - 1, x1)
    img[y0, x0:x1 + 1] = color
    img[y1, x0:x1 + 1] = color
    img[y0:y1 + 1, x0] = color
    img[y0:y1 + 1, x1] = color


def annotate(img, tracks, zone_risk, plant_cfg, frame: int, safety_index: float, scale: int = 2) -> np.ndarray:
    out = img.astype(np.float32)
    zs = plant_cfg.zone_size
    r = np.clip(zone_risk.reshape(plant_cfg.n_rows, plant_cfg.n_cols) / 1.5, 0, 1)
    heat = np.kron(r, np.ones((zs, zs)))[..., None]
    out = out * (1 - 0.35 * heat) + np.array([255, 0, 0]) * 0.35 * heat
    out = np.clip(out, 0, 255).astype(np.uint8)
    out = np.kron(out, np.ones((scale, scale, 1), np.uint8))
    for t in tracks:
        cy, cx = int(t.y * scale), int(t.x * scale)
        if t.cls == "forklift":
            _box(out, cy - 12 * scale // 2, cx - 12 * scale // 2, cy + 12 * scale // 2, cx + 12 * scale // 2, CYAN)
        else:
            col = GREEN if (t.helmet and t.vest) else RED
            h = 7 * scale
            _box(out, cy - h, cx - h, cy + h, cx + h, col)
            _box(out, cy - h - 1, cx - h - 1, cy + h + 1, cx + h + 1, col)
    # zone grid
    for y in range(0, out.shape[0], zs * scale):
        out[y, :] = (out[y, :] * 0.6).astype(np.uint8)
    for x in range(0, out.shape[1], zs * scale):
        out[:, x] = (out[:, x] * 0.6).astype(np.uint8)
    # safety-index bar (top-left)
    w = int(120 * scale * safety_index / 100)
    out[4:10, 4:4 + 120 * scale] = 60
    out[4:10, 4:4 + w] = GREEN if safety_index > 60 else RED
    return out
