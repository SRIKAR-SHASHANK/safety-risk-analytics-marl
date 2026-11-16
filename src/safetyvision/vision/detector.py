"""Classical computer-vision object detector + PPE feature extraction.

Detection pipeline (per frame):
  1. illumination-invariant chromaticity transform  r,g,b / (r+g+b)
  2. colour-prior segmentation (worker clothing, forklift paint)
  3. morphological closing + hole filling
  4. connected-component labelling, area filtering
  5. splitting of merged blobs (k-means on pixel coordinates)

The design is intentionally lightweight (NumPy/SciPy only) so the whole
project runs on a laptop CPU. ``safetyvision.vision.yolo_adapter`` shows how
to swap in a deep detector (e.g. YOLOv8 trained on a PPE dataset) for real
CCTV footage without changing any downstream analytics or MARL code.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

from ..config import VisionConfig
from . import renderer as R


def _chroma(c) -> np.ndarray:
    c = np.asarray(c, np.float32)
    return c / c.sum()


C_VEST, C_SHIRT, C_FORK = _chroma(R.VEST), _chroma(R.SHIRT), _chroma(R.FORKLIFT)
C_HELMET, C_HAIR = _chroma(R.HELMET), _chroma(R.HAIR)
_STRUCT = np.ones((3, 3), bool)


@dataclass
class Detection:
    cls: str          # "person" | "forklift"
    y: float
    x: float
    area: int
    bbox: tuple       # (y0, x0, y1, x1)


class ColorBlobDetector:
    def __init__(self, cfg: VisionConfig):
        self.cfg = cfg

    @staticmethod
    def chroma_image(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        f = img.astype(np.float32)
        s = f.sum(axis=-1, keepdims=True) + 1e-3
        return f / s, s[..., 0]

    def _mask(self, ch: np.ndarray, s: np.ndarray, ref: np.ndarray, min_sum: float) -> np.ndarray:
        d = np.linalg.norm(ch - ref, axis=-1)
        return (d < self.cfg.color_tol) & (s > min_sum)

    @staticmethod
    def _kmeans(points: np.ndarray, k: int, iters: int = 8) -> np.ndarray:
        idx = np.linspace(0, len(points) - 1, k).astype(int)
        cent = points[idx].astype(float)
        for _ in range(iters):
            lab = np.argmin(((points[:, None, :] - cent[None]) ** 2).sum(-1), axis=1)
            for j in range(k):
                if np.any(lab == j):
                    cent[j] = points[lab == j].mean(0)
        return cent

    def detect(self, img: np.ndarray) -> list[Detection]:
        cfg = self.cfg
        ch, s = self.chroma_image(img)
        dets: list[Detection] = []

        # ---- people (hi-vis vest OR plain shirt), heads filled in by hole filling
        person = self._mask(ch, s, C_VEST, 150) | self._mask(ch, s, C_SHIRT, 80)
        person = ndimage.binary_closing(person, _STRUCT)
        person = ndimage.binary_fill_holes(person)
        lab, n = ndimage.label(person)
        if n:
            for i, sl in enumerate(ndimage.find_objects(lab), start=1):
                ys, xs = np.nonzero(lab[sl] == i)
                area = len(ys)
                if area < cfg.min_person_area:
                    continue
                ys = ys + sl[0].start
                xs = xs + sl[1].start
                k = int(round(area / cfg.single_person_area))
                if k >= 2:
                    pts = np.stack([ys, xs], 1).astype(float)
                    for cy, cx in self._kmeans(pts, k):
                        dets.append(Detection("person", float(cy), float(cx), area // k,
                                              (int(cy) - 5, int(cx) - 5, int(cy) + 6, int(cx) + 6)))
                else:
                    dets.append(Detection("person", float(ys.mean()), float(xs.mean()), area,
                                          (sl[0].start, sl[1].start, sl[0].stop, sl[1].stop)))

        # ---- forklifts
        fork = ndimage.binary_opening(self._mask(ch, s, C_FORK, 120), _STRUCT)
        lab, n = ndimage.label(fork)
        if n:
            for i, sl in enumerate(ndimage.find_objects(lab), start=1):
                area = int((lab[sl] == i).sum())
                if area < cfg.min_forklift_area:
                    continue
                cy, cx = ndimage.center_of_mass(lab[sl] == i)
                dets.append(Detection("forklift", float(cy + sl[0].start), float(cx + sl[1].start), area,
                                      (sl[0].start, sl[1].start, sl[0].stop, sl[1].stop)))
        return dets


# --------------------------------------------------------------------------- PPE features
_PATCH = 6  # half-size -> 13x13 patch
_yy, _xx = np.mgrid[-_PATCH:_PATCH + 1, -_PATCH:_PATCH + 1]
_RR = np.sqrt(_yy ** 2 + _xx ** 2)
_HEAD_M = _RR <= 1.6
_TORSO_M = (_RR >= 2.6) & (_RR <= 4.6)


def ppe_features(img: np.ndarray, det: Detection) -> np.ndarray:
    """Hand-crafted colour descriptor of the head and torso regions of a person blob."""
    P = _PATCH
    pad = np.pad(img, ((P, P), (P, P), (0, 0)), mode="edge").astype(np.float32)
    cy, cx = int(round(det.y)) + P, int(round(det.x)) + P
    patch = pad[cy - P:cy + P + 1, cx - P:cx + P + 1]
    s = patch.sum(-1, keepdims=True) + 1e-3
    ch = patch / s
    feats = []
    for m in (_HEAD_M, _TORSO_M):
        c = ch[m]
        feats += list(c.mean(0)) + list(c.std(0)) + [s[m].mean() / 765.0]
        feats += [np.mean(np.linalg.norm(c - ref, axis=1) < 0.12) for ref in (C_HELMET, C_HAIR, C_VEST, C_SHIRT)]
    return np.asarray(feats, np.float32)


def match_detections(dets: list[Detection], gt: list[dict], max_dist: float) -> list[tuple[int, int]]:
    """Hungarian matching on centroid distance; returns (det_idx, gt_idx) pairs."""
    if not dets or not gt:
        return []
    D = np.array([[np.hypot(d.y - g["y"], d.x - g["x"]) for g in gt] for d in dets])
    r, c = linear_sum_assignment(D)
    return [(i, j) for i, j in zip(r, c) if D[i, j] <= max_dist]
