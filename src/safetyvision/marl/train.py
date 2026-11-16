"""Training / evaluation loops for the inspection MARL problem."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from ..config import MARLConfig
from .agents import Policy
from .env import SafetyInspectionEnv

VAL_SEED = 500_000     # validation episodes (model selection)
TEST_SEED = 900_000    # held-out test episodes (reported results)


def evaluate(policy: Policy, env: SafetyInspectionEnv, episodes: int, seed: int = TEST_SEED) -> pd.DataFrame:
    """Greedy (no exploration) evaluation; every policy sees the same episode seeds."""
    rows = []
    for ep in range(episodes):
        obs = env.reset(seed=seed + ep)
        ret, done = 0.0, False
        while not done:
            acts = policy.act(env, obs, explore=False)
            obs, _, _, done, info = env.step(acts)
            ret += info["true_team_reward"]
        rows.append(dict(policy=policy.name, episode=ep, team_return=ret, **env.summary()))
    return pd.DataFrame(rows)


def train(policy: Policy, env: SafetyInspectionEnv, cfg: MARLConfig, seed: int = 0,
          log_every: int = 250, verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Epsilon-greedy training with linear epsilon / learning-rate decay.

    Every ``cfg.val_every`` episodes the greedy policy is scored on a fixed set of
    validation episodes and the best Q-table is kept (early-stopping style model
    selection; test episodes are never used for selection).
    Returns (per-episode training log, validation log).
    """
    rows, val_rows = [], []
    decay_eps = max(1, int(cfg.eps_decay_frac * cfg.episodes))
    best_q, best_val = policy.Q.copy(), -np.inf
    t0 = time.time()
    for ep in range(cfg.episodes):
        frac = min(1.0, ep / decay_eps)
        policy.eps = cfg.eps_start + (cfg.eps_end - cfg.eps_start) * frac
        policy.lr = cfg.lr + (cfg.lr_end - cfg.lr) * ep / max(1, cfg.episodes - 1)
        obs = env.reset(seed=seed * 1_000_003 + ep)
        ret, done = 0.0, False
        while not done:
            acts = policy.act(env, obs, explore=True)
            nobs, r, ri, done, info = env.step(acts)
            policy.update(obs, acts, r, ri, nobs, done)
            obs, ret = nobs, ret + info["true_team_reward"]
        rows.append(dict(episode=ep, eps=policy.eps, team_return=ret, **env.summary()))

        if (ep + 1) % cfg.val_every == 0 or ep + 1 == cfg.episodes:
            v = evaluate(policy, env, cfg.val_episodes, seed=VAL_SEED)
            val_rows.append(dict(episode=ep + 1, val_return=v.team_return.mean(), val_incidents=v.incidents.mean()))
            if v.team_return.mean() > best_val:
                best_val, best_q = v.team_return.mean(), policy.Q.copy()
        if verbose and (ep + 1) % log_every == 0:
            recent = pd.DataFrame(rows[-log_every:])
            vr = val_rows[-1] if val_rows else {"val_return": float("nan")}
            print(f"  [{policy.name}] ep {ep + 1:5d}/{cfg.episodes}  eps={policy.eps:.2f}  "
                  f"train_return={recent.team_return.mean():7.1f}  incidents={recent.incidents.mean():5.1f}  "
                  f"val_return={vr['val_return']:7.1f}  ({time.time() - t0:.0f}s)")
    policy.Q = best_q
    return pd.DataFrame(rows), pd.DataFrame(val_rows)
