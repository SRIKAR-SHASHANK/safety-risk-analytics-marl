"""Reproduce every result of the project.

    python scripts/run_all.py            # full experiment (~25 min on a laptop CPU)
    python scripts/run_all.py --quick    # smoke test (~1 min)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from safetyvision.analytics.predictive import build_dataset, train_and_evaluate  # noqa: E402
from safetyvision.config import ExperimentConfig  # noqa: E402
from safetyvision.marl.agents import IQL, VDN, GreedyPolicy, PatrolPolicy, RandomPolicy  # noqa: E402
from safetyvision.marl.env import SafetyInspectionEnv  # noqa: E402
from safetyvision.marl.train import evaluate, train  # noqa: E402
from safetyvision.pipeline import collect_ppe_dataset, run_shift  # noqa: E402
from safetyvision.reporting import plots  # noqa: E402
from safetyvision.vision.ppe_classifier import PPEClassifier  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "results"))
    args = ap.parse_args()
    cfg = ExperimentConfig.quick() if args.quick else ExperimentConfig()
    out = Path(args.out)
    fig = out / "figures"
    fig.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2, default=str))
    s = cfg.seed
    summary: dict = {}
    T0 = time.time()

    # ------------------------------------------------------------------ 1. PPE classifier
    print("[1/5] Training PPE classifier on detected person crops ...")
    X, yh, yv = collect_ppe_dataset(cfg, seed=s, n_frames=cfg.cv_train_frames)
    Xv, yhv, yvv = collect_ppe_dataset(cfg, seed=s + 7, n_frames=max(300, cfg.cv_train_frames // 3))
    clf = PPEClassifier(seed=s).fit(X, yh, yv)
    ppe_eval = clf.evaluate(Xv, yhv, yvv)
    summary["ppe_classifier"] = dict(n_train=int(len(X)), n_test=int(len(Xv)), **ppe_eval)
    plots.ppe_confusion(ppe_eval, fig / "ppe_confusion.png")
    print(f"      helmet acc={ppe_eval['helmet']['accuracy']:.3f}  vest acc={ppe_eval['vest']['accuracy']:.3f}")

    # ------------------------------------------------------------------ 2. full CV shift
    print(f"[2/5] Running CV + tracking + risk engine over a {cfg.cv_eval_frames}-frame shift ...")
    shift = run_shift(cfg, clf, seed=s + 100, n_frames=cfg.cv_eval_frames, gif_frames=cfg.gif_frames)
    risk, layout = shift["risk"], shift["layout"]
    events, incidents = risk.events_frame(), shift["incidents"]
    events.to_csv(out / "safety_events.csv", index=False)
    incidents.to_csv(out / "ground_truth_incidents.csv", index=False)
    summary["cv_detection"] = shift["metrics"]
    summary["cv_fps"] = shift["fps"]
    summary["events_by_type"] = events.kind.value_counts().to_dict()
    summary["incidents_by_type"] = incidents.kind.value_counts().to_dict()
    grid = risk.risk_grid()
    pd.DataFrame(grid).to_csv(out / "zone_risk_grid.csv", index=False)
    plots.risk_heatmap(grid, layout, fig / "risk_heatmap.png")
    plots.events_over_time(events, cfg.cv_eval_frames, fig / "events_over_time.png")
    plots.safety_index(risk.safety_index, fig / "safety_index.png")
    plots.save_gif(shift["gif"], fig / "cctv_annotated.gif")
    # correlation between CV risk heat-map and true incident locations
    inc_zone = np.bincount(incidents.zone, minlength=cfg.plant.n_zones) if len(incidents) else np.zeros(cfg.plant.n_zones)
    summary["risk_vs_incident_spearman"] = float(pd.Series(risk.cum_risk).corr(pd.Series(inc_zone), method="spearman"))
    m = shift["metrics"]
    print(f"      person F1={m['person']['f1']:.3f} forklift F1={m['forklift']['f1']:.3f} "
          f"violation F1={m['violation']['f1']:.3f}  fps={shift['fps']:.1f}")

    # ------------------------------------------------------------------ 3. predictive analytics
    print("[3/5] Predictive incident model ...")
    df = build_dataset(risk.feature_tensor(), incidents, layout.zone_static_features(), cfg.predict)
    pred = train_and_evaluate(df, cfg.predict, seed=s)
    summary["prediction"] = {k: v for k, v in pred.items() if k != "models"}
    summary["prediction"]["models"] = {k: {kk: vv for kk, vv in v.items() if kk != "roc"} for k, v in pred["models"].items()}
    plots.roc_curves(pred, fig / "roc_incident_forecast.png")
    plots.feature_importance(pred, fig / "feature_importance.png")
    for k, v in pred["models"].items():
        print(f"      {k:40s} ROC-AUC={v['roc_auc']:.3f}  PR-AUC={v['pr_auc']:.3f}")

    # ------------------------------------------------------------------ 4. MARL
    print("[4/5] Multi-agent RL inspector dispatch ...")
    mc = cfg.marl
    env = SafetyInspectionEnv(mc, cfg.plant.n_rows, cfg.plant.n_cols, risk.cum_risk,
                              recall=m["violation"]["recall"], fp_rate=m["violation"]["fp_per_zone_frame"], seed=s)
    logs, val_logs, learners = {}, {}, []
    for cls in (IQL, VDN):
        p = cls(cfg.plant.n_zones, mc.n_agents, mc.gamma, mc.lr, seed=s)
        logs[p.name], val_logs[p.name] = train(p, env, mc, seed=s, log_every=max(1, mc.episodes // 6))
        learners.append(p)
    plots.learning_curves(logs, fig / "marl_learning_curves.png")
    pd.concat([d.assign(policy=k) for k, d in logs.items()]).to_csv(out / "marl_training_log.csv", index=False)
    policies = [RandomPolicy(s), PatrolPolicy(), GreedyPolicy(), *learners]
    res = pd.concat([evaluate(p, env, mc.eval_episodes) for p in policies], ignore_index=True)
    res.to_csv(out / "marl_evaluation.csv", index=False)
    plots.policy_comparison(res, fig / "marl_policy_comparison.png")
    table = res.groupby("policy", sort=False)[["incidents", "resolved", "mean_response_time",
                                               "idle_inspection_rate", "team_return"]].mean()
    summary["marl"] = table.round(3).to_dict(orient="index")
    best_base = table.loc[[p.name for p in policies[:3]], "incidents"].idxmin()
    tests = {}
    for p in learners:
        a, b = res[res.policy == p.name].incidents, res[res.policy == best_base].incidents
        tests[p.name] = dict(vs=best_base, mean_diff=float(a.mean() - b.mean()),
                             p_value=float(mannwhitneyu(a, b).pvalue))
    summary["marl_significance"] = tests
    print(table.round(2).to_string())

    # ------------------------------------------------------------------ 5. report
    print("[5/5] Writing report ...")
    summary["runtime_min"] = (time.time() - T0) / 60
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    write_report(out, summary, table, pred)
    print(f"Done in {summary['runtime_min']:.1f} min -> {out}")


def write_report(out: Path, s: dict, table: pd.DataFrame, pred: dict) -> None:
    d = s["cv_detection"]
    L = ["# Results report", "", "_Auto-generated by `scripts/run_all.py`._", "",
         "## 1. Computer vision", "",
         "| Task | Precision | Recall | F1 |", "|---|---|---|---|"]
    for k, name in (("person", "Person detection"), ("forklift", "Forklift detection"),
                    ("violation", "PPE-violation alert (tracked)")):
        L.append(f"| {name} | {d[k]['precision']:.3f} | {d[k]['recall']:.3f} | {d[k]['f1']:.3f} |")
    pc = s["ppe_classifier"]
    L += ["", f"PPE classifier (per crop): hard-hat accuracy **{pc['helmet']['accuracy']:.3f}**, "
              f"vest accuracy **{pc['vest']['accuracy']:.3f}** on {pc['n_test']} held-out crops. "
              f"Throughput: **{s['cv_fps']:.1f} FPS** on CPU.", "",
          "![PPE confusion](figures/ppe_confusion.png)", "",
          "## 2. Risk analytics", "",
          "Events detected: " + ", ".join(f"{k} = {v}" for k, v in s["events_by_type"].items()) + ".", "",
          "Ground-truth incidents: " + ", ".join(f"{k} = {v}" for k, v in s["incidents_by_type"].items()) + ".", "",
          f"Spearman correlation between the CV risk heat-map and true incident counts per zone: "
          f"**{s['risk_vs_incident_spearman']:.3f}**.", "",
          "![Risk heat-map](figures/risk_heatmap.png)", "", "![Events](figures/events_over_time.png)", "",
          "## 3. Predictive analytics (incident in zone within next 60 frames)", "",
          f"Train rows {pred['n_train']}, test rows {pred['n_test']}, test positive rate {pred['pos_rate_test']:.3f}.", "",
          "| Model | ROC-AUC | PR-AUC | Brier | Lift @ top 10% |", "|---|---|---|---|---|"]
    for k, v in pred["models"].items():
        L.append(f"| {k} | {v['roc_auc']:.3f} | {v['pr_auc']:.3f} | {v['brier']:.4f} | {v['lift_at_top10pct']:.2f}x |")
    L += ["", "![ROC](figures/roc_incident_forecast.png)", "", "![Importance](figures/feature_importance.png)", "",
          "## 4. Multi-agent RL inspector dispatch", "",
          "Mean over held-out test episodes (same seeds for every policy).", "",
          "| Policy | Incidents | Resolved | Response time | Idle-inspection rate | Team return |",
          "|---|---|---|---|---|---|"]
    for name, r in table.iterrows():
        L.append(f"| {name} | {r.incidents:.1f} | {r.resolved:.1f} | {r.mean_response_time:.2f} | "
                 f"{r.idle_inspection_rate:.2f} | {r.team_return:.1f} |")
    L += [""]
    for k, t in s["marl_significance"].items():
        L.append(f"- {k} vs {t['vs']}: mean incident difference {t['mean_diff']:+.1f}, "
                 f"Mann-Whitney p = {t['p_value']:.3g}")
    L += ["", "![Learning curves](figures/marl_learning_curves.png)", "",
          "![Policy comparison](figures/marl_policy_comparison.png)", "",
          f"Total runtime: {s['runtime_min']:.1f} min."]
    (out / "REPORT.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
