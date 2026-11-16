"""Baseline policies and multi-agent RL learners (IQL, VDN).

Both learners use tabular Q-functions over the discrete local observation
``(zone, own-zone hazard level, best direction, teammate-closer flag)`` with
parameter sharing across the homogeneous inspectors (a single table, every
agent conditions on its own observation).

* **IQL** - Independent Q-Learning (Tan, 1993): each agent treats the others
  as part of the environment and learns from its *individual* reward.
* **VDN** - Value-Decomposition Networks (Sunehag et al., 2018): the joint
  action-value is factorised as Q_tot = sum_i Q_i(o_i, a_i) and trained
  end-to-end on the *shared team reward*, which solves the multi-agent credit
  assignment problem and lets agents learn to *not* duplicate each other.
"""
from __future__ import annotations

import numpy as np

from .env import N_ACTIONS, SafetyInspectionEnv


def obs_index(o: tuple) -> int:
    level, best, cb, second, cs, crowded = o
    return ((((level * 5 + best) * 2 + cb) * 5 + second) * 2 + cs) * 2 + crowded


def n_states(n_zones: int | None = None) -> int:
    return 3 * 5 * 2 * 5 * 2 * 2


class Policy:
    name = "policy"
    learns = False

    def act(self, env: SafetyInspectionEnv, obs: list[tuple], explore: bool = False) -> list[int]:
        raise NotImplementedError


class RandomPolicy(Policy):
    name = "Random"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def act(self, env, obs, explore=False):
        return [int(a) for a in self.rng.integers(0, N_ACTIONS, len(obs))]


class PatrolPolicy(Policy):
    """Fixed-route patrol: the plant is split into vertical sectors, one per inspector,
    swept in a boustrophedon pattern; the inspector stops to inspect whenever its
    current zone shows an alert. This mirrors a typical manual safety walk."""

    name = "Fixed patrol"

    def __init__(self):
        self.routes = None
        self.ptr = None

    def _build(self, env):
        n, R, C = env.cfg.n_agents, env.n_rows, env.n_cols
        bounds = np.linspace(0, C, n + 1).round().astype(int)
        self.routes = []
        for i in range(n):
            cols = list(range(bounds[i], bounds[i + 1]))
            route = []
            for k, c in enumerate(cols):
                rows = range(R) if k % 2 == 0 else range(R - 1, -1, -1)
                route += [r * C + c for r in rows]
            route += route[::-1][1:-1]
            self.routes.append(route)
        self.ptr = [0] * n

    def act(self, env, obs, explore=False):
        if self.routes is None or env.t == 0:
            self._build(env)
        acts = []
        for i, o in enumerate(obs):
            z, level = int(env.pos[i]), o[0]
            if level > 0:
                acts.append(0)
                continue
            route = self.routes[i]
            if z == route[self.ptr[i]]:
                self.ptr[i] = (self.ptr[i] + 1) % len(route)
            target = route[self.ptr[i]]
            acts.append(_step_towards(env, z, target))
        return acts


def _step_towards(env, z, target) -> int:
    r, c = divmod(z, env.n_cols)
    tr, tc = divmod(target, env.n_cols)
    if tr < r:
        return 1
    if tr > r:
        return 2
    if tc > c:
        return 3
    if tc < c:
        return 4
    return 0


class GreedyPolicy(Policy):
    """Every inspector chases the highest CV-reported risk around it (no coordination)."""

    name = "Greedy (CV risk-chasing)"

    def act(self, env, obs, explore=False):
        return [0 if level > 0 else best for (level, best, *_rest) in obs]


class _TabularLearner(Policy):
    learns = True

    def __init__(self, n_zones: int, n_agents: int, gamma: float, lr: float, seed: int = 0):
        self.Q = np.zeros((n_states(n_zones), N_ACTIONS))
        self.n_agents = n_agents
        self.gamma, self.lr = gamma, lr
        self.eps = 1.0
        self.rng = np.random.default_rng(seed)

    def act(self, env, obs, explore=False):
        acts = []
        for o in obs:
            if explore and self.rng.random() < self.eps:
                acts.append(int(self.rng.integers(N_ACTIONS)))
            else:
                q = self.Q[obs_index(o)]
                acts.append(int(self.rng.choice(np.flatnonzero(q == q.max()))))
        return acts


class IQL(_TabularLearner):
    name = "IQL (independent)"

    def update(self, obs, acts, team_r, indiv_r, next_obs, done):
        for i in range(self.n_agents):
            s, s2 = obs_index(obs[i]), obs_index(next_obs[i])
            target = indiv_r[i] + (0.0 if done else self.gamma * self.Q[s2].max())
            self.Q[s, acts[i]] += self.lr * (target - self.Q[s, acts[i]])


class VDN(_TabularLearner):
    name = "VDN (cooperative MARL)"

    def update(self, obs, acts, team_r, indiv_r, next_obs, done):
        s = [obs_index(o) for o in obs]
        s2 = [obs_index(o) for o in next_obs]
        q_tot = sum(self.Q[s[i], acts[i]] for i in range(self.n_agents))
        q_next = 0.0 if done else sum(self.Q[s2[i]].max() for i in range(self.n_agents))
        delta = team_r + self.gamma * q_next - q_tot
        step = self.lr * delta          # gradient of 0.5*delta^2 w.r.t. each Q_i
        for i in range(self.n_agents):
            self.Q[s[i], acts[i]] += step
