"""Multi-agent safety-inspection environment.

``n_agents`` mobile safety officers (or inspection drones) move on the plant's
zone grid. Hazards (PPE violations, unsafe acts, blocked exits, ...) appear in
zones as a Poisson process whose zone-level rates are **learned from the
computer-vision risk analytics** (cumulative CV risk heat-map). Unresolved
hazards age and escalate into incidents with increasing probability.

Agents see the plant through the CCTV/CV system, so their hazard map is
imperfect: each true hazard is detected with probability ``recall`` and each
zone produces spurious alerts at rate ``fp_rate`` -- both calibrated from the
measured performance of the CV pipeline.

Actions per agent: 0 = inspect current zone, 1..4 = move N/S/E/W.
Only agents that *inspect* resolve hazards (up to ``inspect_capacity``).
"""
from __future__ import annotations

import numpy as np

from ..config import MARLConfig

MOVES = np.array([(0, 0), (-1, 0), (1, 0), (0, 1), (0, -1)])
N_ACTIONS = 5


class SafetyInspectionEnv:
    def __init__(self, cfg: MARLConfig, n_rows: int, n_cols: int, zone_prior: np.ndarray,
                 recall: float = 0.9, fp_rate: float = 0.01, seed: int = 0):
        self.cfg = cfg
        self.n_rows, self.n_cols = n_rows, n_cols
        self.n_zones = n_rows * n_cols
        prior = np.asarray(zone_prior, float).ravel()
        prior = prior / prior.sum() if prior.sum() > 0 else np.ones(self.n_zones) / self.n_zones
        self.base_rate = cfg.hazard_rate * (0.1 / self.n_zones + 0.9 * prior)
        self.prior_map = cfg.prior_weight * self.base_rate      # what the CV analytics says is risky
        self.recall, self.fp_rate = float(recall), float(fp_rate)
        self.rng = np.random.default_rng(seed)
        self._build_cones()
        self.reset(seed)

    # ------------------------------------------------------------------ geometry
    def _build_cones(self) -> None:
        """W[d] (Z x Z): distance-weighted look-ahead cones used for local observations."""
        R = self.cfg.obs_radius
        Z = self.n_zones
        rc = np.array([divmod(z, self.n_cols) for z in range(Z)])
        dr = rc[None, :, 0] - rc[:, None, 0]
        dc = rc[None, :, 1] - rc[:, None, 1]
        dist = np.abs(dr) + np.abs(dc)
        self.dist = dist
        within = (dist > 0) & (dist <= R)
        cones = [
            np.eye(Z, dtype=bool),
            within & (dr < 0) & (np.abs(dc) <= -dr),   # N
            within & (dr > 0) & (np.abs(dc) <= dr),    # S
            within & (dc > 0) & (np.abs(dr) <= dc),    # E
            within & (dc < 0) & (np.abs(dr) <= -dc),   # W
        ]
        w = np.where(dist > 0, 1.0 / np.maximum(dist, 1), 1.5)
        self.cones = cones
        self.W = np.stack([c * w for c in cones])     # (5, Z, Z)

    def _move(self, z: int, a: int) -> int:
        r, c = divmod(z, self.n_cols)
        r2, c2 = r + MOVES[a][0], c + MOVES[a][1]
        if 0 <= r2 < self.n_rows and 0 <= c2 < self.n_cols:
            return int(r2 * self.n_cols + c2)
        return z

    # ------------------------------------------------------------------ core API
    def reset(self, seed: int | None = None) -> list[tuple]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        cfg = self.cfg
        self.t = 0
        noise = self.rng.lognormal(0.0, cfg.rate_noise, self.n_zones)
        self.rate = self.base_rate * noise * (self.base_rate.sum() / (self.base_rate * noise).sum())
        self.phase = self.rng.uniform(0, 2 * np.pi)
        self.hazards: list[list[int]] = [[] for _ in range(self.n_zones)]   # ages
        self.pos = self.rng.choice(self.n_zones, cfg.n_agents, replace=False).astype(int)
        self.stats = dict(incidents=0, resolved=0, spawned=0, response_times=[], idle_inspections=0,
                          inspections=0, visited=set(self.pos.tolist()))
        self._observe_map()
        return self.observations()

    def _observe_map(self) -> None:
        true = np.array([len(h) for h in self.hazards])
        detected = self.rng.binomial(true, self.recall)
        false_alarms = self.rng.poisson(self.fp_rate, self.n_zones)
        self.obs_map = (detected + false_alarms).astype(float)

    def _closer_flag(self, i: int, z: int, d: int) -> int:
        """1 if a teammate is at least as close to the most attractive zone in direction d."""
        cand = np.where(self.cones[d][z])[0]
        vals = self.look_map[cand] * self.W[d][z, cand]
        if len(cand) == 0 or vals.max() <= 0:
            return 0
        target = int(cand[np.argmax(vals)])
        others = np.delete(self.pos, i)
        if d == 0:
            return int(np.any(others == z))
        return int(np.any(self.dist[others, target] <= self.dist[z, target]))

    def observations(self) -> list[tuple]:
        """Per-agent discrete local observation derived from the shared CV hazard map:

        (own-zone level 0/1/2+, best direction, teammate-closer-to-best flag,
         second-best direction, teammate-closer-to-second flag, crowded flag)

        Directions: 0 = here, 1..4 = N/S/E/W look-ahead cones (distance-weighted).
        The observation is zone-agnostic so a single policy generalises over the plant.
        """
        self.look_map = self.obs_map + self.prior_map
        scores = self.W @ self.look_map           # (5, Z)
        obs = []
        for i, z in enumerate(self.pos):
            s = scores[:, z]
            order = np.argsort(-s, kind="stable")
            best = int(order[0]) if s[order[0]] > 0 else 0
            second = int(order[1]) if s[order[1]] > 0 else best
            level = int(min(self.obs_map[z], 2))
            others = np.delete(self.pos, i)
            crowded = int(np.any(self.dist[others, z] <= 1))
            obs.append((level, best, self._closer_flag(i, z, best), second,
                        self._closer_flag(i, z, second), crowded))
        return obs

    def step(self, actions) -> tuple[list[tuple], float, np.ndarray, bool, dict]:
        cfg, rng = self.cfg, self.rng
        n = cfg.n_agents
        indiv = np.zeros(n)
        resolved_total, moves = 0, 0
        idle_before = self.stats["idle_inspections"]
        # 1) movement / inspection
        for i, a in enumerate(actions):
            if a == 0:
                z = self.pos[i]
                self.stats["inspections"] += 1
                k = min(cfg.inspect_capacity, len(self.hazards[z]))
                if k == 0:
                    self.stats["idle_inspections"] += 1
                    indiv[i] -= cfg.idle_inspect_cost
                else:
                    self.hazards[z].sort(reverse=True)          # oldest first
                    done = self.hazards[z][:k]
                    self.hazards[z] = self.hazards[z][k:]
                    self.stats["response_times"].extend(done)
                    resolved_total += k
                    indiv[i] += k * cfg.resolve_reward
            else:
                self.pos[i] = self._move(self.pos[i], a)
                indiv[i] -= cfg.move_cost
                moves += 1
                self.stats["visited"].add(int(self.pos[i]))
        # 2) escalation of unresolved hazards
        incidents, expected_incidents = 0, 0.0
        for z in range(self.n_zones):
            if not self.hazards[z]:
                continue
            ages = np.array(self.hazards[z]) + 1
            p = np.minimum(cfg.base_escalation + cfg.escalation_growth * ages, cfg.max_escalation)
            esc = rng.random(len(ages)) < p
            expected_incidents += float(p.sum())
            incidents += int(esc.sum())
            self.hazards[z] = ages[~esc].tolist()
        # 3) new hazards (shift-dependent intensity)
        shift = 1.0 + 0.5 * np.sin(2 * np.pi * self.t / cfg.horizon + self.phase)
        new = rng.poisson(self.rate * shift)
        for z in np.nonzero(new)[0]:
            self.hazards[z].extend([0] * int(new[z]))
        self.stats["spawned"] += int(new.sum())
        self.stats["incidents"] += incidents
        self.stats["resolved"] += resolved_total
        idle = self.stats["idle_inspections"] - idle_before
        action_cost = moves * cfg.move_cost + idle * cfg.idle_inspect_cost
        true_r = resolved_total * cfg.resolve_reward - incidents * cfg.incident_cost - action_cost
        # Learning signal: replace the sampled incident count by its conditional expectation
        # (sum of escalation probabilities). Same expected return, far lower variance
        # (Rao-Blackwellisation) -> much easier credit assignment for the learners.
        inc_term = expected_incidents if cfg.expected_incident_reward else incidents
        team_r = resolved_total * cfg.resolve_reward - inc_term * cfg.incident_cost - action_cost
        indiv -= inc_term * cfg.incident_cost / n
        self.t += 1
        self._observe_map()
        done = self.t >= cfg.horizon
        return self.observations(), float(team_r), indiv, done, {"true_team_reward": float(true_r)}

    def summary(self) -> dict:
        s = self.stats
        return dict(
            incidents=s["incidents"], resolved=s["resolved"], spawned=s["spawned"],
            mean_response_time=float(np.mean(s["response_times"])) if s["response_times"] else float("nan"),
            idle_inspection_rate=s["idle_inspections"] / max(1, s["inspections"]),
            coverage=len(s["visited"]) / self.n_zones,
            backlog=int(sum(len(h) for h in self.hazards)),
        )
