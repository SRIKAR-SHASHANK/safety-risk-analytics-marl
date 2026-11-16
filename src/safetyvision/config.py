"""Central configuration for every stage of the pipeline.

All experiment knobs live here so that a run is fully described by the
dataclasses below (they are dumped to ``results/config.json`` on every run).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass
class PlantConfig:
    """Synthetic manufacturing plant (top-down CCTV view)."""

    height: int = 240               # frame height in pixels
    width: int = 320                # frame width in pixels
    zone_size: int = 40             # plant is split into zone_size x zone_size cells -> 6 x 8 zones
    n_workers: int = 18
    n_forklifts: int = 3
    p_no_helmet: float = 0.18       # initial probability a worker is missing a helmet
    p_no_vest: float = 0.15         # initial probability a worker is missing a hi-vis vest
    ppe_drop_rate: float = 0.002    # per-frame prob. a compliant worker removes PPE (x risk propensity)
    ppe_fix_rate: float = 0.01      # per-frame prob. a non-compliant worker puts PPE back on
    danger_margin: int = 10         # width (px) of the hazard band painted around each machine

    @property
    def n_rows(self) -> int:
        return self.height // self.zone_size

    @property
    def n_cols(self) -> int:
        return self.width // self.zone_size

    @property
    def n_zones(self) -> int:
        return self.n_rows * self.n_cols


@dataclass
class RenderConfig:
    """Camera degradations applied to every rendered frame."""

    noise_std: float = 9.0                    # additive Gaussian sensor noise
    illumination_range: tuple = (0.75, 1.15)  # global brightness multiplier
    blur_prob: float = 0.3                    # prob. of motion/defocus blur
    blur_sigma: float = 0.8
    occlusion_prob: float = 0.08              # prob. that steam / pillars occlude part of the frame


@dataclass
class VisionConfig:
    color_tol: float = 0.11         # chromaticity distance tolerance for colour segmentation
    min_person_area: int = 18
    single_person_area: int = 81    # area of one rendered worker (disk r=5)
    min_forklift_area: int = 50
    match_dist: float = 6.0         # px, detection <-> ground-truth matching radius
    track_max_dist: float = 12.0
    track_max_missed: int = 5
    ppe_vote_window: int = 5        # temporal majority vote for PPE state per track


@dataclass
class RiskConfig:
    near_miss_px: float = 16.0
    w_no_helmet: float = 1.0
    w_no_vest: float = 0.8
    w_danger_zone: float = 1.5
    w_restricted: float = 2.0
    w_near_miss: float = 3.0
    ema_alpha: float = 0.05
    event_min_frames: int = 3       # a condition must persist this long before it is logged as an event


@dataclass
class PredictConfig:
    history: int = 30               # frames of history used to build features
    horizon: int = 60               # predict "incident in zone within next horizon frames"
    stride: int = 10
    train_frac: float = 0.7


@dataclass
class MARLConfig:
    n_agents: int = 4
    episodes: int = 2000
    horizon: int = 200
    eval_episodes: int = 200
    gamma: float = 0.95
    lr: float = 0.05
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_frac: float = 0.7
    hazard_rate: float = 0.9        # expected new hazards per step (plant-wide)
    rate_noise: float = 0.35        # per-episode log-normal perturbation of zone hazard rates
    base_escalation: float = 0.01   # per-step prob. an unresolved hazard becomes an incident
    escalation_growth: float = 0.004
    max_escalation: float = 0.25
    inspect_capacity: int = 3       # hazards an inspector can resolve per step
    incident_cost: float = 10.0
    resolve_reward: float = 1.0
    move_cost: float = 0.05
    idle_inspect_cost: float = 0.1  # inspecting an empty zone
    prior_weight: float = 15.0      # weight of the CV risk prior in the agents' look-ahead map
    lr_end: float = 0.01
    val_every: int = 100            # validation-based checkpoint selection
    val_episodes: int = 10
    expected_incident_reward: bool = True  # low-variance reward shaping for learning
    obs_radius: int = 12             # look-ahead radius (zones) over the shared CCTV hazard map


@dataclass
class ExperimentConfig:
    seed: int = 42
    cv_train_frames: int = 1200
    cv_eval_frames: int = 6000
    gif_frames: int = 80
    plant: PlantConfig = field(default_factory=PlantConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    predict: PredictConfig = field(default_factory=PredictConfig)
    marl: MARLConfig = field(default_factory=MARLConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def quick(cls) -> "ExperimentConfig":
        """Small configuration for smoke tests / CI (runs in well under a minute)."""
        cfg = cls()
        cfg.cv_train_frames = 300
        cfg.cv_eval_frames = 900
        cfg.gif_frames = 20
        cfg.marl.episodes = 150
        cfg.marl.eval_episodes = 20
        cfg.marl.horizon = 100
        return cfg
