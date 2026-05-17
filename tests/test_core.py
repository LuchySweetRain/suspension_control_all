from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import load_config
from controllers import MPCController, PIDController, SPDFController
from controllers.rl_transformer import RLTransformerController
from envs import HalfCarEnv
from models import HalfCarModel, HalfCarParams
from rl.ddpg import DDPGConfig
from rl.networks import TransformerActor, TransformerGaussianActor, TransformerValue
from rl.ppo import PPOConfig, finish_ppo_trajectory
from rl.replay_buffer import MixedReplaySampler, ReplayBuffer
from rl.sac import SACConfig
from rl.td3 import TD3Config
from roads.road_profiles import RoadProfileFactory


def cfg():
    c = load_config(ROOT / "configs" / "default.yaml")
    c["episode_seconds"] = 1.0
    c["mpc"]["max_iter"] = 2
    return c


def test_half_car_zero_equilibrium():
    model = HalfCarModel(HalfCarParams.from_seed(42))
    x = np.zeros(8)
    road = np.zeros(4)
    action = np.zeros(2)
    dx = model.derivative(x, action, road)
    assert np.allclose(dx, 0.0)


def test_half_car_parameters_match_spdf_constant_sim():
    p = HalfCarParams()
    assert p.mb == 1200.0
    assert p.Ip == 600.0
    assert p.mwf == 100.0
    assert p.mwr == 100.0
    assert p.kf1 == 15000.0
    assert p.kr1 == 15000.0
    assert p.knf1 == 1000.0
    assert p.knr1 == 1000.0
    assert p.be == 1500.0
    assert p.bc == 1200.0
    assert p.kf2 == 200000.0
    assert p.kr2 == 200000.0
    assert p.bf2 == 1500.0
    assert p.br2 == 2000.0
    assert p.a == 1.2
    assert p.b == 1.5
    assert p.fmax == 5000.0


def test_road_preview_shape():
    c = cfg()
    scenario = c["scenarios"][0]
    road = RoadProfileFactory.create(scenario, duration=1.0, dt=c["dt"])
    preview = road.preview(0.0, scenario["speed"], c["wheelbase"], c["preview"]["steps"], c["control_dt"])
    assert preview.shape == (10, 2)
    assert np.all(np.isfinite(preview))


def test_pid_spdf_outputs_without_preview():
    c = cfg()
    env = HalfCarEnv(c, scenario=c["scenarios"][0], use_preview=False)
    obs, info = env.reset()
    for controller in (
        PIDController(env.params, env.control_dt, env.force_limit),
        SPDFController(env.params, env.control_dt),
    ):
        action = controller.compute_action(obs, info)
        assert action.shape == (2,)
        assert np.all(np.isfinite(action))
        assert np.max(np.abs(action)) <= env.force_limit


def test_mpc_uses_preview_shape():
    c = cfg()
    env = HalfCarEnv(c, scenario=c["scenarios"][0], use_preview=True)
    obs, info = env.reset()
    assert info["road_preview"].shape == (10, 2)
    action = MPCController(env.model, c).compute_action(obs, info)
    assert action.shape == (2,)
    assert np.all(np.isfinite(action))


def test_transformer_actor_output_shape():
    c = cfg()
    td3_cfg = TD3Config.from_project_config(c)
    actor = TransformerActor(td3_cfg)
    obs = torch.zeros(4, td3_cfg.obs_dim)
    action = actor(obs)
    assert action.shape == (4, 2)


def test_transformer_gaussian_actor_and_value_shapes():
    c = cfg()
    sac_cfg = SACConfig.from_project_config(c)
    actor = TransformerGaussianActor(sac_cfg)
    value = TransformerValue(PPOConfig.from_project_config(c))
    obs = torch.zeros(4, sac_cfg.obs_dim)
    action, logp = actor.sample(obs)
    assert action.shape == (4, 2)
    assert logp.shape == (4, 1)
    assert value(obs).shape == (4, 1)


def test_rl_controller_algorithms_smoke():
    c = cfg()
    env = HalfCarEnv(c, scenario=c["scenarios"][0], use_preview=True)
    obs, info = env.reset()
    for algorithm in ("td3", "ddpg", "sac", "ppo"):
        controller = RLTransformerController(c, algorithm=algorithm)
        action = controller.compute_action(obs, info)
        assert action.shape == (2,)
        assert np.all(np.isfinite(action))
        assert np.max(np.abs(action)) <= env.force_limit


def test_ddpg_config_matches_td3_observation_space():
    c = cfg()
    td3_cfg = TD3Config.from_project_config(c)
    ddpg_cfg = DDPGConfig.from_project_config(c)
    assert ddpg_cfg.obs_dim == td3_cfg.obs_dim
    assert ddpg_cfg.act_dim == td3_cfg.act_dim


def test_ppo_advantage_finish_shapes():
    returns, advantages = finish_ppo_trajectory(
        rewards=[1.0, 1.0],
        values=[0.5, 0.25],
        dones=[False, True],
        gamma=0.99,
        lam=0.95,
    )
    assert len(returns) == 2
    assert len(advantages) == 2
    assert np.all(np.isfinite(returns))
    assert np.all(np.isfinite(advantages))


def test_controller_smoke_1s():
    c = cfg()
    for scenario in c["scenarios"]:
        for controller_cls, use_preview in (
            (PIDController, False),
            (SPDFController, False),
            (MPCController, True),
        ):
            env = HalfCarEnv(c, scenario=scenario, use_preview=use_preview)
            obs, info = env.reset()
            if controller_cls is PIDController:
                controller = PIDController(env.params, env.control_dt, env.force_limit)
            elif controller_cls is SPDFController:
                controller = SPDFController(env.params, env.control_dt)
            else:
                controller = MPCController(env.model, c)
            done = False
            while not done:
                action = controller.compute_action(obs if use_preview else obs[: env.base_obs_dim], info)
                obs, _, terminated, truncated, info = env.step(action)
                assert np.all(np.isfinite(obs))
                done = terminated or truncated


def test_mixed_replay_sampler_decays_expert_ratio():
    online = ReplayBuffer(obs_dim=3, act_dim=2, capacity=20)
    expert = ReplayBuffer(obs_dim=3, act_dim=2, capacity=20)
    for i in range(10):
        obs = np.ones(3, dtype=np.float32) * i
        act = np.ones(2, dtype=np.float32)
        online.store(obs, act, 0.0, obs, False)
        expert.store(obs, act, 1.0, obs, False)
    sampler = MixedReplaySampler(online, expert, 0.5, 0.1, 10)
    sampler.set_episode(0)
    assert np.isclose(sampler.expert_ratio, 0.5)
    sampler.set_episode(10)
    assert np.isclose(sampler.expert_ratio, 0.1)
    batch = sampler.sample(8, torch.device("cpu"))
    assert batch["obs"].shape == (8, 3)
    assert batch["act"].shape == (8, 2)
