import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from safetyvision.config import ExperimentConfig, MARLConfig  # noqa: E402
from safetyvision.marl.agents import IQL, VDN, GreedyPolicy, PatrolPolicy, n_states, obs_index  # noqa: E402
from safetyvision.marl.env import SafetyInspectionEnv  # noqa: E402
from safetyvision.simulation.factory import FactorySimulator  # noqa: E402
from safetyvision.vision.detector import ColorBlobDetector, match_detections  # noqa: E402
from safetyvision.vision.renderer import Renderer  # noqa: E402

CFG = ExperimentConfig()


def test_simulator_keeps_agents_in_frame():
    sim = FactorySimulator(CFG.plant, seed=0)
    for _ in range(200):
        st = sim.step()
    for w in st.workers:
        assert 0 <= w["y"] < CFG.plant.height and 0 <= w["x"] < CFG.plant.width
    assert len(st.forklifts) == CFG.plant.n_forklifts


def test_detector_finds_workers_on_clean_frame():
    sim = FactorySimulator(CFG.plant, seed=1)
    ren = Renderer(sim.layout, CFG.render, seed=1)
    st = sim.step()
    dets = [d for d in ColorBlobDetector(CFG.vision).detect(ren.render_clean(st)) if d.cls == "person"]
    matched = match_detections(dets, st.workers, CFG.vision.match_dist)
    assert len(matched) >= 0.8 * len(st.workers)


def _env(seed=0):
    cfg = MARLConfig(horizon=30)
    return SafetyInspectionEnv(cfg, 6, 8, np.ones(48), recall=0.9, fp_rate=0.01, seed=seed), cfg


def test_env_step_and_observation_space():
    env, cfg = _env()
    obs = env.reset(seed=3)
    assert len(obs) == cfg.n_agents
    for _ in range(cfg.horizon):
        obs, r, ri, done, info = env.step([0, 1, 2, 3])
        assert all(0 <= obs_index(o) < n_states() for o in obs)
        assert ri.shape == (cfg.n_agents,)
    assert done
    s = env.summary()
    assert s["resolved"] + s["incidents"] + s["backlog"] == s["spawned"]


def test_baselines_and_learners_run():
    env, cfg = _env()
    for p in (GreedyPolicy(), PatrolPolicy(), IQL(48, cfg.n_agents, 0.95, 0.1), VDN(48, cfg.n_agents, 0.95, 0.1)):
        obs = env.reset(seed=1)
        done = False
        while not done:
            a = p.act(env, obs, explore=True)
            nobs, r, ri, done, _ = env.step(a)
            if getattr(p, "learns", False):
                p.update(obs, a, r, ri, nobs, done)
            obs = nobs
        if getattr(p, "learns", False):
            assert np.abs(p.Q).sum() > 0
