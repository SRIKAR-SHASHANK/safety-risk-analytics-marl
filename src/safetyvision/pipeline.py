"""End-to-end perception -> analytics pipeline over a simulated shift."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .analytics.risk import RiskEngine
from .config import ExperimentConfig
from .reporting.annotate import annotate
from .simulation.factory import FactorySimulator
from .vision.detector import ColorBlobDetector, match_detections, ppe_features
from .vision.ppe_classifier import PPEClassifier
from .vision.renderer import Renderer
from .vision.tracker import CentroidTracker


def collect_ppe_dataset(cfg: ExperimentConfig, seed: int, n_frames: int, stride: int = 3):
    """Detect people on rendered frames and label each detection from matched ground truth."""
    sim = FactorySimulator(cfg.plant, seed=seed)
    ren = Renderer(sim.layout, cfg.render, seed=seed + 1)
    det = ColorBlobDetector(cfg.vision)
    X, yh, yv = [], [], []
    for f in range(n_frames):
        state = sim.step()
        if f % stride:
            continue
        img = ren.render(state)
        persons = [d for d in det.detect(img) if d.cls == "person"]
        for i, j in match_detections(persons, state.workers, cfg.vision.match_dist):
            X.append(ppe_features(img, persons[i]))
            yh.append(state.workers[j]["helmet"])
            yv.append(state.workers[j]["vest"])
    return np.array(X), np.array(yh), np.array(yv)


def _prf(tp, fp, fn) -> dict:
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    return dict(precision=p, recall=r, f1=2 * p * r / max(p + r, 1e-9), tp=int(tp), fp=int(fp), fn=int(fn))


def run_shift(cfg: ExperimentConfig, clf: PPEClassifier, seed: int, n_frames: int,
              gif_frames: int = 0, verbose: bool = True) -> dict:
    """Simulate one shift, run CV + tracking + risk engine, and score everything against ground truth."""
    sim = FactorySimulator(cfg.plant, seed=seed)
    ren = Renderer(sim.layout, cfg.render, seed=seed + 1)
    det = ColorBlobDetector(cfg.vision)
    trk = CentroidTracker(cfg.vision)
    risk = RiskEngine(sim.layout, cfg.risk)
    counts = {k: [0, 0, 0] for k in ("person", "forklift", "violation")}
    incidents, gif, gt_viol_zone = [], [], np.zeros(cfg.plant.n_zones)
    t0 = time.time()
    for f in range(n_frames):
        state = sim.step()
        img = ren.render(state)
        dets = det.detect(img)
        persons = [d for d in dets if d.cls == "person"]
        forks = [d for d in dets if d.cls == "forklift"]
        if persons:
            ph, pv = clf.predict(np.stack([ppe_features(img, d) for d in persons]))
        else:
            ph, pv = np.zeros(0, bool), np.zeros(0, bool)
        ppe = list(zip(ph, pv)) + [(True, True)] * len(forks)
        tracks = trk.update(persons + forks, ppe)
        risk.update(f, tracks)
        incidents += [dict(frame=i.frame, zone=i.zone, kind=i.kind, worker_id=i.worker_id) for i in state.incidents]

        # ---- detection metrics (frame-level, centroid matching)
        for cls, d_list, gt in (("person", persons, state.workers), ("forklift", forks, state.forklifts)):
            m = len(match_detections(d_list, gt, cfg.vision.match_dist if cls == "person" else 10.0))
            counts[cls][0] += m
            counts[cls][1] += len(d_list) - m
            counts[cls][2] += len(gt) - m
        # ---- PPE-violation alert metrics (track-level, after temporal voting)
        ptracks = [t for t in tracks if t.cls == "person"]
        gt_v = [w for w in state.workers if not (w["helmet"] and w["vest"])]
        for w in gt_v:
            gt_viol_zone[sim.layout.zone_of(w["y"], w["x"])] += 1
        pr_v = [t for t in ptracks if not (t.helmet and t.vest)]
        m = len(match_detections(pr_v, gt_v, cfg.vision.match_dist)) if pr_v and gt_v else 0
        counts["violation"][0] += m
        counts["violation"][1] += len(pr_v) - m
        counts["violation"][2] += len(gt_v) - m

        if f < gif_frames:
            gif.append(annotate(img, tracks, risk.zone_risk, cfg.plant, f, risk.safety_index[-1]))
        if verbose and (f + 1) % 1000 == 0:
            print(f"  frame {f + 1}/{n_frames}  tracks={len(tracks)}  events={len(risk.events)}  "
                  f"({time.time() - t0:.0f}s)")

    metrics = {k: _prf(*v) for k, v in counts.items()}
    metrics["violation"]["fp_per_zone_frame"] = counts["violation"][1] / (n_frames * cfg.plant.n_zones)
    return dict(
        metrics=metrics,
        risk=risk,
        incidents=pd.DataFrame(incidents, columns=["frame", "zone", "kind", "worker_id"]),
        gt_violation_zone=gt_viol_zone,
        layout=sim.layout,
        gif=gif,
        fps=n_frames / (time.time() - t0),
    )
