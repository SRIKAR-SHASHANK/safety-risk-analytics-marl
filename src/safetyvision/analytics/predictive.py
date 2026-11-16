"""Predictive risk analytics: will an incident happen in zone z in the next H frames?

Features are built **only from computer-vision output** (risk-engine feature
tensor) plus static layout descriptors; labels come from the simulator's
hidden ground-truth incident log. A chronological train/test split avoids
temporal leakage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..config import PredictConfig
from .risk import FEATURES

STATIC = ["static_hazard_frac", "static_restricted_frac", "static_aisle_frac"]


def build_dataset(feat: np.ndarray, incidents: pd.DataFrame, static: np.ndarray,
                  cfg: PredictConfig) -> pd.DataFrame:
    """feat: (T, Z, F) tensor from RiskEngine; incidents: DataFrame[frame, zone]."""
    T, Z, F = feat.shape
    inc = np.zeros((T, Z))
    for fr, z in zip(incidents["frame"], incidents["zone"]):
        inc[fr, z] += 1
    csum = np.concatenate([np.zeros((1, Z, F)), np.cumsum(feat, axis=0)])
    icum = np.concatenate([np.zeros((1, Z)), np.cumsum(inc, axis=0)])
    rows = []
    for t in range(cfg.history, T - cfg.horizon, cfg.stride):
        hist = (csum[t] - csum[t - cfg.history]) / cfg.history          # mean over history window
        recent = (csum[t] - csum[t - 5]) / 5                              # last 5 frames
        past_inc = icum[t] - icum[max(0, t - 300)]
        future = icum[t + cfg.horizon] - icum[t]
        for z in range(Z):
            rows.append([t, z, *hist[z], *recent[z], past_inc[z], *static[z], int(future[z] > 0)])
    cols = (["frame", "zone"] + [f"mean_{f}" for f in FEATURES] + [f"recent_{f}" for f in FEATURES]
            + ["past_incidents_300"] + STATIC + ["label"])
    return pd.DataFrame(rows, columns=cols)


def train_and_evaluate(df: pd.DataFrame, cfg: PredictConfig, seed: int = 0) -> dict:
    cut = df["frame"].quantile(cfg.train_frac)
    tr, te = df[df["frame"] <= cut], df[df["frame"] > cut]
    feat_cols = [c for c in df.columns if c not in ("frame", "zone", "label")]
    Xtr, ytr, Xte, yte = tr[feat_cols].values, tr["label"].values, te[feat_cols].values, te["label"].values

    models = {
        "Density baseline (LogReg)": (make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
                                      ["mean_n_person", "mean_n_forklift"]),
        "CV risk features (LogReg)": (make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5)),
                                      feat_cols),
        "CV risk features (Gradient Boosting)": (HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0,
            random_state=seed), feat_cols),
    }
    out = {"n_train": int(len(tr)), "n_test": int(len(te)),
           "pos_rate_train": float(ytr.mean()), "pos_rate_test": float(yte.mean()), "models": {}}
    best, best_auc = None, -1
    for name, (model, cols) in models.items():
        idx = [feat_cols.index(c) for c in cols]
        model.fit(Xtr[:, idx], ytr)
        p = model.predict_proba(Xte[:, idx])[:, 1]
        auc = roc_auc_score(yte, p) if 0 < yte.sum() < len(yte) else float("nan")
        fpr, tpr, _ = roc_curve(yte, p)
        # precision among the top-10% highest-risk zone-windows ("inspection budget" view)
        k = max(1, int(0.1 * len(p)))
        top = np.argsort(-p)[:k]
        out["models"][name] = dict(
            roc_auc=float(auc), pr_auc=float(average_precision_score(yte, p)),
            brier=float(brier_score_loss(yte, p)),
            precision_at_top10pct=float(yte[top].mean()),
            lift_at_top10pct=float(yte[top].mean() / max(yte.mean(), 1e-9)),
            roc=(fpr.tolist(), tpr.tolist()),
        )
        if auc > best_auc:
            best, best_auc = (name, model, idx), auc
    name, model, idx = best
    imp = permutation_importance(model, Xte[:, idx], yte, scoring="roc_auc", n_repeats=5, random_state=seed)
    out["best_model"] = name
    out["importance"] = sorted(zip([feat_cols[i] for i in idx], imp.importances_mean.tolist()),
                               key=lambda t: -t[1])
    return out
