"""Matplotlib figures for the report."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ORANGE, BLUE, DARK, GREY = "#E07B1F", "#2F6FD6", "#16191F", "#9AA0A6"
PALETTE = [GREY, "#7F8C8D", ORANGE, "#5DA5DA", BLUE]


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def risk_heatmap(grid: np.ndarray, layout, path, title="Cumulative CV risk per zone"):
    fig, ax = plt.subplots(figsize=(8, 6))
    zs = layout.cfg.zone_size
    im = ax.imshow(grid, cmap="YlOrRd", extent=(0, grid.shape[1] * zs, grid.shape[0] * zs, 0))
    for m in layout.machines:
        ax.add_patch(plt.Rectangle((m.x0, m.y0), m.x1 - m.x0, m.y1 - m.y0, fill=False, ec=DARK, lw=1.5))
    for r in layout.restricted:
        ax.add_patch(plt.Rectangle((r.x0, r.y0), r.x1 - r.x0, r.y1 - r.y0, fill=False, ec=BLUE, lw=1.5, ls="--"))
    ax.axhspan(96, 144, color=BLUE, alpha=0.06)
    fig.colorbar(im, ax=ax, label="summed severity")
    ax.set_title(title + "\n(black = machines, dashed blue = restricted store, band = forklift aisle)")
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    _save(fig, path)


def events_over_time(events: pd.DataFrame, n_frames: int, path, bucket=300):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    if len(events):
        events = events.assign(bucket=events.frame // bucket)
        tab = events.pivot_table(index="bucket", columns="kind", values="frame", aggfunc="count", fill_value=0)
        tab = tab.reindex(range(n_frames // bucket + 1), fill_value=0)
        tab.index = tab.index * bucket
        tab.plot(ax=ax, marker="o", ms=3)
    ax.set_xlabel("frame")
    ax.set_ylabel(f"events per {bucket} frames")
    ax.set_title("Safety events detected by the CV risk engine over the shift")
    _save(fig, path)


def safety_index(si, path):
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(si, color=ORANGE, lw=1)
    ax.set_ylim(0, 100)
    ax.set_xlabel("frame")
    ax.set_ylabel("Plant Safety Index")
    ax.set_title("Live plant safety index (100 = no active risk)")
    _save(fig, path)


def ppe_confusion(ppe_eval: dict, path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, name in zip(axes, ("helmet", "vest")):
        cm = np.array(ppe_eval[name]["confusion"])
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xticks([0, 1], ["missing", "worn"])
        ax.set_yticks([0, 1], ["missing", "worn"])
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(f"{name}: acc {ppe_eval[name]['accuracy']:.3f}")
    _save(fig, path)


def roc_curves(pred: dict, path):
    fig, ax = plt.subplots(figsize=(6, 6))
    for (name, m), c in zip(pred["models"].items(), [GREY, BLUE, ORANGE]):
        fpr, tpr = m["roc"]
        ax.plot(fpr, tpr, color=c, label=f"{name} (AUC {m['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], ls=":", color="k")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("Zone incident forecast (next 60 frames)")
    ax.legend(fontsize=8, loc="lower right")
    _save(fig, path)


def feature_importance(pred: dict, path, top=12):
    imp = pred["importance"][:top][::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh([k for k, _ in imp], [v for _, v in imp], color=ORANGE)
    ax.set_xlabel("permutation importance (drop in ROC-AUC)")
    ax.set_title(f"Top features - {pred['best_model']}")
    _save(fig, path)


def learning_curves(logs: dict, path, window=50):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for (name, df), c in zip(logs.items(), [BLUE, ORANGE]):
        axes[0].plot(df.episode, df.team_return.rolling(window, min_periods=1).mean(), color=c, label=name)
        axes[1].plot(df.episode, df.incidents.rolling(window, min_periods=1).mean(), color=c, label=name)
    axes[0].set_title("Training team return (moving avg)")
    axes[1].set_title("Incidents per episode during training (moving avg)")
    for ax in axes:
        ax.set_xlabel("episode")
        ax.legend()
    _save(fig, path)


def policy_comparison(res: pd.DataFrame, path):
    g = res.groupby("policy", sort=False)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, col, title in zip(axes, ["incidents", "resolved", "team_return"],
                              ["Incidents / episode (lower is better)", "Hazards resolved / episode",
                               "Team return (higher is better)"]):
        m, s = g[col].mean(), g[col].std() / np.sqrt(g.size())
        ax.bar(range(len(m)), m.values, yerr=1.96 * s.values, color=PALETTE[:len(m)], capsize=4)
        ax.set_xticks(range(len(m)), [n.split(" (")[0] for n in m.index], rotation=20)
        ax.set_title(title)
    _save(fig, path)


def save_gif(frames, path, fps=8):
    from PIL import Image
    if not frames:
        return
    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=int(1000 / fps), loop=0)
