from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import load_config
from controllers import MPCController, PIDController, SPDFController
from controllers.rl_transformer import RLTransformerController
from controllers.prior import residual_gate, shield_policy_action, shield_residual_action
from envs import HalfCarEnv, MuJoCoFullCarEnv, MuJoCoHalfCarEnv, MuJoCoVehicleEnv
from experiments.evaluation import evaluate_all, make_controller
from scripts.benchmark_vector_env import benchmark_vector_env
from scripts.export_mujoco_env_spec import export_env_spec
from scripts.generate_road_dataset import generate_dataset
from scripts.import_road_directory import import_road_directory
from scripts.preflight_mujoco_training import preflight_mujoco_training
from scripts.check_gym_registration import check_registration
from scripts.collect_env_statistics import collect_env_statistics
from scripts.collect_expert_dataset import collect_expert_dataset
from scripts.run_mujoco_benchmark import run_benchmark
from scripts.run_mujoco_robustness_matrix import run_robustness_matrix
from scripts.run_projection_seed_sweep import run_projection_seed_sweep
from scripts.run_si_rppo_ablation import build_claim_report, run_si_rppo_ablation
from scripts.summarize_benchmark import build_report
from scripts.train_rl import ScenarioSampler
from scripts.train_rl import pretrain_ppo_from_imitation
from scripts.validate_training_config import validate_config
from scripts.validate_mujoco_env import validate_environment
from models import HalfCarModel, HalfCarParams
from rl.ddpg import DDPGConfig
from rl.networks import TransformerActor, TransformerGaussianActor, TransformerValue
from rl.ppo import PPOConfig, finish_ppo_trajectory
from rl.ppo import PPOAgent
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


def test_extended_road_profiles_smoke():
    c = cfg()
    scenarios = [
        {"type": "sine", "speed": 10.0, "amplitude": 0.02, "wavelength": 3.0},
        {"type": "step", "height": 0.03, "start_time": 0.2},
        {"type": "pothole", "speed": 10.0, "depth": 0.04, "length": 0.5},
        {"type": "table", "times": [0.0, 0.5, 1.0], "heights": [0.0, 0.02, -0.01]},
        {
            "type": "composite",
            "speed": 10.0,
            "components": [
                {"type": "sine", "amplitude": 0.01, "wavelength": 2.0},
                {"type": "bump", "height": 0.02, "length": 0.3},
            ],
        },
    ]
    for scenario in scenarios:
        road = RoadProfileFactory.create(scenario, duration=1.0, dt=c["dt"])
        preview = road.preview(0.1, scenario.get("speed", 10.0), c["wheelbase"], 5, c["control_dt"])
        assert preview.shape == (5, 2)
        assert np.all(np.isfinite(preview))


def test_csv_road_profile_smoke(tmp_path):
    road_file = tmp_path / "road.csv"
    road_file.write_text("0.0,0.0\n0.5,0.02\n1.0,-0.01\n", encoding="utf-8")
    c = cfg()
    road = RoadProfileFactory.create({"type": "csv", "path": str(road_file)}, duration=1.0, dt=c["dt"])
    preview = road.preview(0.1, 10.0, c["wheelbase"], 5, c["control_dt"])
    assert preview.shape == (5, 2)
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


def test_ppo_unsafe_penalty_accepts_trajectory_flags():
    c = cfg()
    c["rl"]["ppo"] = {
        "train_epochs": 1,
        "minibatch_size": 2,
        "projection_penalty_weight": 0.1,
        "unsafe_penalty_weight": 0.5,
        "action_delta_penalty_weight": 0.25,
    }
    ppo_cfg = PPOConfig.from_project_config(c)
    agent = PPOAgent(ppo_cfg, torch.device("cpu"))
    trajectory = {
        "obs": np.zeros((2, ppo_cfg.obs_dim), dtype=np.float32),
        "act": np.zeros((2, ppo_cfg.act_dim), dtype=np.float32),
        "logp": [0.0, 0.0],
        "ret": [0.0, -1.0],
        "adv": [0.5, -0.5],
        "projection_error": [0.0, 0.2],
        "unsafe": [0.0, 1.0],
        "action_delta": [0.0, 0.1],
    }
    value_loss, actor_loss = agent.train_trajectory(trajectory)
    assert np.isfinite(value_loss)
    assert np.isfinite(actor_loss)
    assert ppo_cfg.unsafe_penalty_weight == 0.5
    assert ppo_cfg.action_delta_penalty_weight == 0.25


def test_mujoco_half_car_env_render_smoke():
    c = cfg()
    c["episode_seconds"] = 0.05
    env = MuJoCoHalfCarEnv(c, scenario=c["scenarios"][0], use_preview=True, width=160, height=90)
    obs, _ = env.reset()
    obs, reward, terminated, truncated, _ = env.step(np.zeros(2, dtype=np.float32))
    frame = env.render()
    env.close()
    assert obs.shape == (env.obs_dim,)
    assert np.isfinite(reward)
    assert not terminated
    assert frame.shape == (90, 160, 3)
    assert frame.dtype == np.uint8


def test_mujoco_vehicle_env_dynamics_smoke():
    c = cfg()
    c["episode_seconds"] = 0.05
    c["mujoco"] = {"settle_seconds": 0.3}
    env = MuJoCoVehicleEnv(c, scenario=c["scenarios"][0], use_preview=True, width=160, height=90)
    obs, info = env.reset()
    assert obs.shape == (env.obs_dim,)
    assert info["engine"] == "mujoco"
    assert np.max(np.abs(obs[:12])) < 1e-6
    obs, reward, terminated, truncated, info = env.step(np.zeros(2, dtype=np.float32))
    frame = env.render()
    env.close()
    assert obs.shape == (env.obs_dim,)
    assert np.isfinite(reward)
    assert not terminated
    assert "derived" in info
    assert frame.shape == (90, 160, 3)
    assert frame.dtype == np.uint8


def test_mujoco_full_car_env_dynamics_smoke():
    c = load_config(ROOT / "configs" / "mujoco_full_car.yaml")
    c["episode_seconds"] = 0.05
    c["mujoco"]["settle_seconds"] = 0.3
    env = MuJoCoFullCarEnv(c, scenario=c["scenarios"][0], use_preview=True, width=160, height=90)
    obs, info = env.reset()
    assert obs.shape == (env.obs_dim,)
    assert info["engine"] == "mujoco_full_car"
    assert info["road_corners"].shape == (8,)
    assert np.max(np.abs(obs[:14])) < 1e-6
    obs, reward, terminated, truncated, info = env.step(np.zeros(2, dtype=np.float32))
    frame = env.render()
    env.close()
    assert obs.shape == (env.obs_dim,)
    assert np.isfinite(reward)
    assert not terminated
    assert "ddroll" in info["derived"]
    assert frame.shape == (90, 160, 3)
    assert frame.dtype == np.uint8


def test_gymnasium_registration_full_car_smoke():
    pytest = __import__("pytest")
    gym = pytest.importorskip("gymnasium")
    report = check_registration(config_path=ROOT / "configs" / "mujoco_full_car_corner.yaml", smoke=False)
    assert report["gymnasium_available"]
    env = gym.make(
        "ActiveSuspensionMuJoCoFullCar-v0",
        config_path=str(ROOT / "configs" / "mujoco_full_car_corner.yaml"),
        scenario_index=0,
        width=160,
        height=90,
    )
    obs, info = env.reset(seed=11)
    obs, reward, terminated, truncated, info = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
    env.close()
    assert obs.shape == env.observation_space.shape
    assert info["engine"] == "mujoco_full_car"
    assert np.isfinite(reward)
    assert not terminated


def test_mujoco_full_car_corner_action_interface():
    c = load_config(ROOT / "configs" / "mujoco_full_car_corner.yaml")
    c["episode_seconds"] = 0.05
    c["mujoco"]["settle_seconds"] = 0.3
    env = MuJoCoFullCarEnv(c, scenario=c["scenarios"][0], use_preview=True, width=160, height=90)
    obs, info = env.reset()
    sac_cfg = SACConfig.from_project_config(c)
    assert env.action_space.shape == (4,)
    assert env.preview_token_dim == 4
    assert info["road_preview"].shape == (env.preview_steps, 4)
    assert obs.shape == (sac_cfg.obs_dim,)
    assert sac_cfg.act_dim == 4
    action = np.asarray([100.0, -100.0, 50.0, -50.0], dtype=np.float32)
    obs, reward, terminated, _, info = env.step(action)
    env.close()
    assert obs.shape == (sac_cfg.obs_dim,)
    assert np.allclose(info["command_corner_action"], action)
    assert info["corner_action"].shape == (4,)
    assert np.isfinite(reward)
    assert not terminated
    assert "reward_components" in info
    assert "action_metrics" in info
    assert "action_delta_rms" in info["action_metrics"]


def test_mujoco_full_car_actuator_dynamics_lag_command():
    c = load_config(ROOT / "configs" / "mujoco_full_car_corner.yaml")
    c["episode_seconds"] = 0.05
    c["settle_seconds"] = 0.0
    c["mujoco"]["settle_seconds"] = 0.2
    c["domain_randomization"]["enabled"] = False
    c["actuator"] = {"enabled": True, "time_constant": 0.05, "rate_limit": 2000.0}
    env = MuJoCoFullCarEnv(c, scenario=c["scenarios"][0], use_preview=True, width=160, height=90)
    env.reset(seed=3)
    command = np.asarray([1000.0, -1000.0, 500.0, -500.0], dtype=np.float32)
    _, _, _, _, info = env.step(command)
    env.close()
    actual = info["corner_action"]
    assert np.allclose(info["command_corner_action"], command)
    assert np.max(np.abs(actual)) < np.max(np.abs(command))
    assert info["actuator"]["enabled"]


def test_mujoco_full_car_configurable_safety_limits_terminate():
    c = load_config(ROOT / "configs" / "mujoco_full_car_corner.yaml")
    c["episode_seconds"] = 0.05
    c["settle_seconds"] = 0.0
    c["mujoco"]["settle_seconds"] = 0.2
    c["domain_randomization"]["enabled"] = False
    c["safety_limits"] = {
        "max_suspension_travel": 0.25,
        "max_pitch": -1.0,
        "max_roll": 0.45,
        "max_wheel_displacement": 1.0,
        "terminate_on_unsafe": True,
        "unsafe_penalty": 10.0,
    }
    env = MuJoCoFullCarEnv(c, scenario=c["scenarios"][0], use_preview=True, width=160, height=90)
    env.reset(seed=5)
    _, _, terminated, _, info = env.step(np.zeros(4, dtype=np.float32))
    env.close()
    assert terminated
    assert info["unsafe"]
    assert "pitch" in info["safety"]["violations"]
    assert info["safety"]["limits"]["terminate_on_unsafe"]


def test_mujoco_full_car_domain_randomization_changes_reset_params():
    c = load_config(ROOT / "configs" / "mujoco_full_car_corner.yaml")
    c["episode_seconds"] = 0.05
    c["mujoco"]["settle_seconds"] = 0.2
    env = MuJoCoFullCarEnv(c, scenario=c["scenarios"][0], use_preview=True, width=160, height=90)
    _, info1 = env.reset(seed=123)
    _, info2 = env.reset()
    env.close()
    dr1 = info1["domain_randomization"]
    dr2 = info2["domain_randomization"]
    assert dr1["enabled"]
    changed = any(
        not np.isclose(dr1[name], dr2[name])
        for name in ("speed", "road_scale", "mb", "Ip", "roll_inertia")
    )
    assert changed


def test_mujoco_full_car_preview_error_model_changes_preview():
    c = load_config(ROOT / "configs" / "mujoco_full_car_corner.yaml")
    c["episode_seconds"] = 0.05
    c["mujoco"]["settle_seconds"] = 0.2
    c["domain_randomization"]["enabled"] = False
    c["preview_error"] = {
        "enabled": True,
        "delay_steps": 0,
        "height_noise_std": 0.0,
        "bias_std": 0.01,
        "dropout_prob": 0.0,
        "scale_error_std": 0.0,
    }
    env = MuJoCoFullCarEnv(c, scenario=c["scenarios"][0], use_preview=True, width=160, height=90)
    obs, info = env.reset(seed=7)
    env.close()
    preview = info["road_preview"]
    clean = info["road_preview_clean"]
    assert obs.shape == (env.obs_dim,)
    assert preview.shape == clean.shape == (env.preview_steps, 4)
    assert info["preview_error"]["enabled"]
    assert not np.allclose(preview, clean)


def test_export_mujoco_env_spec_contains_reproducibility_metadata(tmp_path):
    spec = export_env_spec(
        config_path=ROOT / "configs" / "mujoco_full_car_corner.yaml",
        out_path=tmp_path / "env_spec.json",
    )
    meta = spec["environment_metadata"]
    assert (tmp_path / "env_spec.json").is_file()
    assert meta["engine"] == "mujoco_full_car"
    assert meta["act_dim"] == 4
    assert meta["preview_token_dim"] == 4
    assert meta["full_car"]["action_mode"] == "corner"
    assert "mb_real" in meta["vehicle_params"]
    assert meta["actuator"]["enabled"]
    assert meta["preview_error"]["enabled"]


def test_mujoco_full_car_evaluation_smoke(tmp_path):
    c = load_config(ROOT / "configs" / "mujoco_full_car_corner.yaml")
    c["episode_seconds"] = 0.05
    c["mujoco"]["settle_seconds"] = 0.2
    c["scenarios"] = [c["scenarios"][0]]
    c["evaluation"] = {"controllers": ["PASSIVE"]}
    metrics, trajectories = evaluate_all(c, checkpoints=None, result_dir=tmp_path)
    manifest = json.loads((tmp_path / "evaluation_manifest.json").read_text(encoding="utf-8"))
    assert (tmp_path / "metrics.csv").is_file()
    assert (tmp_path / "evaluation_manifest.json").is_file()
    assert len(metrics) == 1
    assert "RollAccRMS_radps2" in metrics.columns
    assert "MaxSusp_fl_m" in metrics.columns
    assert "PASSIVE_class_b_correlated" in trajectories
    assert manifest["environment_metadata"][0]["engine"] == "mujoco_full_car"


def test_road_dataset_and_benchmark_report_smoke(tmp_path):
    dataset_dir = tmp_path / "dataset"
    manifest = generate_dataset(dataset_dir, duration=0.2, dt=0.01, base_config=ROOT / "configs" / "mujoco_full_car_corner.yaml")
    assert len(manifest["roads"]) >= 3
    assert (dataset_dir / "road_dataset_manifest.json").is_file()
    config = load_config(dataset_dir / "mujoco_full_car_dataset.yaml")
    config["episode_seconds"] = 0.05
    config["mujoco"]["settle_seconds"] = 0.1
    config["scenarios"] = config["scenarios"][:1]
    eval_dir = tmp_path / "eval"
    evaluate_all(config, checkpoints=None, result_dir=eval_dir)
    report = build_report(eval_dir)
    text = report.read_text(encoding="utf-8")
    assert "MuJoCo Suspension Benchmark Report" in text
    assert "Mean Metrics By Controller" in text


def test_import_road_directory_smoke(tmp_path):
    road_dir = tmp_path / "source_roads"
    road_dir.mkdir()
    (road_dir / "road_a.csv").write_text("time,height\n0.0,0.0\n0.1,0.01\n0.2,0.0\n", encoding="utf-8")
    (road_dir / "road_b.csv").write_text("time,height\n0.0,0.0\n0.1,-0.02\n0.2,0.0\n", encoding="utf-8")
    out_dir = tmp_path / "imported"
    manifest = import_road_directory(
        road_dir=road_dir,
        out_dir=out_dir,
        base_config=ROOT / "configs" / "mujoco_full_car_corner.yaml",
        speed=12.0,
    )
    config_path = Path(manifest["config"])
    assert config_path.is_file()
    assert (out_dir / "imported_road_manifest.json").is_file()
    assert len(manifest["roads"]) == 2
    config = load_config(config_path)
    assert len(config["scenarios"]) == 2
    assert config["scenarios"][0]["type"] == "csv"
    assert config["scenarios"][0]["speed"] == 12.0


def test_mujoco_benchmark_runner_passive_smoke(tmp_path):
    out_dir = tmp_path / "benchmark"
    manifest = run_benchmark(
        config_path=ROOT / "configs" / "mujoco_full_car_corner.yaml",
        out_dir=out_dir,
        algorithms=[],
        episodes=0,
        generate_roads=True,
        road_dataset_dir=tmp_path / "roads",
        road_duration=0.2,
        road_dt=0.01,
    )
    assert (out_dir / "benchmark_manifest.json").is_file()
    assert Path(manifest["benchmark_report"]).is_file()
    assert (out_dir / "evaluation" / "metrics.csv").is_file()
    assert manifest["metric_rows"] >= 1


def test_validate_mujoco_env_smoke(tmp_path):
    report = validate_environment(
        config_path=ROOT / "configs" / "mujoco_full_car_corner.yaml",
        out_path=tmp_path / "validation.json",
        max_steps=3,
        action_mode="passive",
        render=False,
    )
    assert report["passed"]
    assert (tmp_path / "validation.json").is_file()
    assert report["total_steps"] >= 3
    assert report["steps_per_second"] > 0.0


def test_validate_training_config_static_checks(tmp_path):
    report = validate_config(ROOT / "configs" / "mujoco_full_car_corner.yaml")
    assert report["passed"]
    assert report["expected_act_dim"] == 4
    assert report["expected_preview_token_dim"] == 4
    bad = load_config(ROOT / "configs" / "mujoco_full_car_corner.yaml")
    bad["rl"]["act_dim"] = 2
    bad_path = tmp_path / "bad.yaml"
    bad_path.write_text(__import__("yaml").safe_dump(bad, sort_keys=False), encoding="utf-8")
    bad_report = validate_config(bad_path, check_roads=False)
    assert not bad_report["passed"]
    assert any("rl.act_dim" in error for error in bad_report["errors"])


def test_vector_env_benchmark_smoke(tmp_path):
    report = benchmark_vector_env(
        config_path=ROOT / "configs" / "mujoco_full_car_corner.yaml",
        out_path=tmp_path / "vector.json",
        num_envs=2,
        steps=2,
        action_mode="passive",
    )
    assert report["passed"]
    assert report["total_transitions"] == 4
    assert report["transitions_per_second"] > 0.0
    assert (tmp_path / "vector.json").is_file()


def test_collect_env_statistics_smoke(tmp_path):
    report = collect_env_statistics(
        config_path=ROOT / "configs" / "mujoco_full_car_corner.yaml",
        out_path=tmp_path / "stats.json",
        episodes=1,
        max_steps=3,
        action_mode="passive",
    )
    assert report["total_steps"] == 3
    assert report["observation"]["count"] == 3
    assert len(report["observation"]["mean"]) == 54
    assert report["reward"]["count"] == 3
    assert "action_metrics" in report
    assert "reward_components" in report
    assert "saturation_ratio" in report["action_metrics"]
    assert (tmp_path / "stats.json").is_file()


def test_mujoco_full_car_history_observation_config():
    c = load_config(ROOT / "configs" / "mujoco_full_car_history.yaml")
    c["episode_seconds"] = 0.05
    c["mujoco"]["settle_seconds"] = 0.2
    c["domain_randomization"]["enabled"] = False
    env = MuJoCoFullCarEnv(c, scenario=c["scenarios"][0], use_preview=True, width=160, height=90)
    obs, info = env.reset(seed=9)
    env.close()
    assert env.history_steps == 5
    assert env.obs_base_dim == 124
    assert obs.shape == (env.obs_dim,)
    assert info["engine"] == "mujoco_full_car"
    assert info["base_obs"].shape == (124,)
    assert SACConfig.from_project_config(c).obs_dim == env.obs_dim


def test_reduced_full_car_preview_controller_smoke():
    c = load_config(ROOT / "configs" / "mujoco_full_car_residual.yaml")
    c["episode_seconds"] = 0.05
    c["mujoco"]["settle_seconds"] = 0.2
    c["domain_randomization"]["enabled"] = False
    env = MuJoCoFullCarEnv(c, scenario=c["scenarios"][0], use_preview=True, width=160, height=90)
    obs, info = env.reset(seed=10)
    controller = make_controller("FULL_CAR_MPC_LITE", env, c)
    action = controller.compute_action(obs, info)
    env.close()
    assert action.shape == env.action_space.shape
    assert np.all(np.isfinite(action))
    assert np.max(np.abs(action)) <= env.force_limit


def test_residual_gate_and_deviation_reward_smoke():
    c = load_config(ROOT / "configs" / "mujoco_full_car_residual.yaml")
    c["episode_seconds"] = 0.05
    c["mujoco"]["settle_seconds"] = 0.2
    c["domain_randomization"]["enabled"] = False
    c["reward"]["deviation"] = 1.0
    env = MuJoCoFullCarEnv(c, scenario=c["scenarios"][0], use_preview=True, width=160, height=90)
    obs, info = env.reset(seed=12)
    cfg = c["residual_control"]
    gate_nominal = residual_gate(info, cfg)
    noisy_info = dict(info)
    noisy_info["preview_error"] = {
        "delay_steps": 3,
        "height_noise_std": 0.02,
        "bias_std": 0.01,
        "dropout_prob": 0.2,
        "scale_error_std": 0.2,
    }
    gate_noisy = residual_gate(noisy_info, cfg)
    near_limit_info = dict(info)
    near_limit_info["safety"] = dict(info["safety"])
    near_limit_info["safety"]["max_abs_suspension_travel"] = 0.99 * near_limit_info["safety"]["limits"]["max_suspension_travel"]
    shielded = shield_residual_action(np.full(env.action_space.shape, env.force_limit, dtype=np.float32), env, near_limit_info, cfg)
    policy_limited = shield_policy_action(
        np.full(env.action_space.shape, env.force_limit, dtype=np.float32),
        env,
        info,
        {"enabled": False, "max_action_fraction": 0.75, "max_delta_fraction": 0.08},
    )
    policy_blocked = shield_policy_action(
        np.full(env.action_space.shape, env.force_limit, dtype=np.float32),
        env,
        near_limit_info,
        {"enabled": True, "max_action_fraction": 0.75, "hard_margin": 0.05, "soft_margin": 0.18},
    )
    prior = np.zeros(env.action_space.shape, dtype=np.float32)
    action = np.full(env.action_space.shape, 100.0, dtype=np.float32)
    _, _, _, _, no_prior_info = env.step(action)
    assert not no_prior_info["has_prior_action"]
    assert no_prior_info["action_metrics"]["deviation_rms"] == 0.0
    assert no_prior_info["reward_components"]["deviation"] == 0.0
    _, reward, _, _, info = env.step({"action": action, "prior_action": prior})
    env.close()
    assert gate_noisy < gate_nominal
    assert np.max(np.abs(shielded)) < 1e-3
    assert np.max(np.abs(policy_limited - env.last_action)) <= 0.08 * env.force_limit + 1e-5
    assert np.max(np.abs(policy_blocked)) < 1e-3
    assert info["has_prior_action"]
    assert np.allclose(info["prior_action"], prior)
    assert info["action_metrics"]["deviation_rms"] > 0.0
    assert info["reward_components"]["deviation"] > 0.0
    assert np.isfinite(reward)


def test_collect_expert_dataset_and_ppo_bc_smoke(tmp_path):
    cfg_path = ROOT / "configs" / "mujoco_full_car_residual.yaml"
    out_path = tmp_path / "expert.npz"
    manifest = collect_expert_dataset(
        config_path=cfg_path,
        out_path=out_path,
        expert="FULL_CAR_MPC_LITE",
        residual_prior="FULL_CAR_MPC_LITE",
        episodes=1,
        max_steps=3,
        seed=21,
    )
    assert out_path.is_file()
    assert Path(str(out_path).replace(".npz", ".manifest.json")).is_file()
    assert manifest["transitions"] == 3
    assert manifest["raw_transitions"] == 3
    assert not manifest["skip_unsafe"]
    assert manifest["skipped_unsafe"] >= 0
    assert 0.0 <= manifest["raw_unsafe_fraction"] <= 1.0
    data = np.load(out_path, allow_pickle=True)
    assert data["obs"].shape[0] == 3
    assert data["act"].shape == data["residual_act"].shape
    assert np.max(np.abs(data["residual_act"])) < 1e-5

    c = load_config(cfg_path)
    c["episode_seconds"] = 0.05
    c["mujoco"]["settle_seconds"] = 0.2
    c["domain_randomization"]["enabled"] = False
    c["imitation"] = {
        "enabled": True,
        "dataset": str(out_path),
        "epochs": 1,
        "batch_size": 2,
        "residual_targets": True,
    }
    agent = PPOAgent(PPOConfig.from_project_config(c), torch.device("cpu"))
    losses = pretrain_ppo_from_imitation(agent, c, tmp_path)
    assert len(losses) == 1
    assert np.isfinite(losses[0])
    assert agent.bc_anchor_obs is None
    assert (tmp_path / "imitation_pretrain.json").is_file()

    c["imitation"]["anchor_enabled"] = True
    c["imitation"]["anchor_max_samples"] = 2
    anchored = PPOAgent(PPOConfig.from_project_config(c), torch.device("cpu"))
    pretrain_ppo_from_imitation(anchored, c, tmp_path)
    assert anchored.bc_anchor_obs is not None
    assert anchored.bc_anchor_obs.shape[0] == 2


def test_si_rppo_ablation_dry_run_writes_variants(tmp_path):
    manifest = run_si_rppo_ablation(
        base_config_path=ROOT / "configs" / "mujoco_full_car_safe_ppo.yaml",
        out_dir=tmp_path / "ablation",
        episodes=1,
        expert_episodes=1,
        expert_max_steps=2,
        expert_controller="PASSIVE",
        skip_unsafe_expert=True,
        train_scenario_limit=1,
        eval_scenario_limit=1,
        episode_seconds=0.05,
        mujoco_settle_seconds=0.2,
        baseline_algorithms=["td3", "sac"],
        dry_run=True,
    )
    assert manifest["dry_run"]
    assert Path(manifest["materialized_base_config"]).is_file()
    materialized = load_config(Path(manifest["materialized_base_config"]))
    assert materialized["episode_seconds"] == 0.05
    assert materialized["mujoco"]["settle_seconds"] == 0.2
    assert len(materialized["scenarios"]) == 1
    assert manifest["expert_manifest"]["planned"]
    assert manifest["expert_controller"] == "PASSIVE"
    assert manifest["skip_unsafe_expert"]
    assert manifest["expert_manifest"]["expert"] == "PASSIVE"
    assert manifest["expert_manifest"]["skip_unsafe"]
    assert set(manifest["variants"]) == {
        "ppo_scratch",
        "bc_ppo",
        "residual_bc_ppo",
        "safe_residual_bc_ppo",
    }
    for report in manifest["variants"].values():
        config_path = Path(report["config"])
        assert config_path.is_file()
        assert report["planned_train_command"]
    residual_cfg = load_config(Path(manifest["variants"]["residual_bc_ppo"]["config"]))
    safe_cfg = load_config(Path(manifest["variants"]["safe_residual_bc_ppo"]["config"]))
    bc_cfg = load_config(Path(manifest["variants"]["bc_ppo"]["config"]))
    scratch_cfg = load_config(Path(manifest["variants"]["ppo_scratch"]["config"]))
    assert scratch_cfg["rl"]["ppo"]["projection_penalty_weight"] == 0.0
    assert scratch_cfg["rl"]["ppo"]["unsafe_penalty_weight"] == 0.0
    assert scratch_cfg["rl"]["ppo"]["action_delta_penalty_weight"] == 0.0
    assert bc_cfg["rl"]["ppo"]["projection_penalty_weight"] > 0.0
    assert bc_cfg["rl"]["ppo"]["unsafe_penalty_weight"] > 0.0
    assert bc_cfg["rl"]["ppo"]["action_delta_penalty_weight"] > 0.0
    assert not residual_cfg["imitation"]["anchor_enabled"]
    assert not residual_cfg["residual_control"]["shield"]["enabled"]
    assert not safe_cfg["imitation"]["anchor_enabled"]
    assert safe_cfg["residual_control"]["shield"]["enabled"]
    assert bc_cfg["imitation"]["anchor_enabled"]
    assert set(manifest["off_policy_baselines"]) == {"td3_baseline", "sac_baseline"}
    for report in manifest["off_policy_baselines"].values():
        config_path = Path(report["config"])
        assert config_path.is_file()
        assert report["planned_train_command"]
    assert Path(manifest["combined_metrics"]).is_file()
    assert manifest["claim_report"]["status"] == "missing_data"
    assert manifest["claim_report"]["core_status"] == "missing_data"
    assert Path(manifest["claim_report"]["json"]).is_file()
    assert Path(manifest["claim_report"]["markdown"]).is_file()


def test_si_rppo_claim_report_detects_supported_ablation(tmp_path):
    rows = [
        {
            "Variant": "ppo_scratch",
            "Controller": "PPO",
            "Scenario": "road_a",
            "EpisodeReturn": -100.0,
            "UnsafeSteps": 2,
            "ActuatorSaturationRatio": 0.2,
            "ActionDeltaRMS_N": 50.0,
            "BodyAccRMS_mps2": 1.2,
            "PitchAccRMS_radps2": 0.3,
            "RollAccRMS_radps2": 0.4,
            "ActionDeviationRMS_N": 20.0,
        },
        {
            "Variant": "bc_ppo",
            "Controller": "PPO",
            "Scenario": "road_a",
            "EpisodeReturn": -80.0,
            "UnsafeSteps": 1,
            "ActuatorSaturationRatio": 0.1,
            "ActionDeltaRMS_N": 40.0,
            "BodyAccRMS_mps2": 1.0,
            "PitchAccRMS_radps2": 0.25,
            "RollAccRMS_radps2": 0.35,
            "ActionDeviationRMS_N": 18.0,
        },
        {
            "Variant": "residual_bc_ppo",
            "Controller": "PPO",
            "Scenario": "road_a",
            "EpisodeReturn": -70.0,
            "UnsafeSteps": 1,
            "ActuatorSaturationRatio": 0.1,
            "ActionDeltaRMS_N": 42.0,
            "BodyAccRMS_mps2": 0.9,
            "PitchAccRMS_radps2": 0.22,
            "RollAccRMS_radps2": 0.32,
            "ActionDeviationRMS_N": 16.0,
        },
        {
            "Variant": "safe_residual_bc_ppo",
            "Controller": "PPO",
            "Scenario": "road_a",
            "EpisodeReturn": -68.0,
            "UnsafeSteps": 0,
            "ActuatorSaturationRatio": 0.02,
            "ActionDeltaRMS_N": 30.0,
            "BodyAccRMS_mps2": 0.92,
            "PitchAccRMS_radps2": 0.23,
            "RollAccRMS_radps2": 0.33,
            "ActionDeviationRMS_N": 8.0,
        },
        {
            "Variant": "td3_baseline",
            "Controller": "TD3",
            "Scenario": "road_a",
            "EpisodeReturn": -90.0,
            "UnsafeSteps": 1,
            "ActuatorSaturationRatio": 0.12,
            "ActionDeltaRMS_N": 60.0,
            "BodyAccRMS_mps2": 1.1,
            "PitchAccRMS_radps2": 0.3,
            "RollAccRMS_radps2": 0.4,
            "ActionDeviationRMS_N": 0.0,
        },
        {
            "Variant": "sac_baseline",
            "Controller": "SAC",
            "Scenario": "road_a",
            "EpisodeReturn": -88.0,
            "UnsafeSteps": 1,
            "ActuatorSaturationRatio": 0.1,
            "ActionDeltaRMS_N": 55.0,
            "BodyAccRMS_mps2": 1.05,
            "PitchAccRMS_radps2": 0.28,
            "RollAccRMS_radps2": 0.38,
            "ActionDeviationRMS_N": 0.0,
        },
    ]
    report = build_claim_report(pd.DataFrame(rows), tmp_path)
    assert report["status"] == "ready"
    statuses = {item["name"]: item["status"] for item in report["comparisons"]}
    assert statuses["projection_aware_imitation"] == "supported"
    assert statuses["imitation_initialization"] == "supported"
    assert statuses["residual_prior_structure"] == "supported"
    assert statuses["safe_residual_gate"] == "supported"
    assert statuses["safe_residual_ppo_vs_td3"] == "supported"
    assert statuses["safe_residual_ppo_vs_sac"] == "supported"
    assert Path(report["json_path"]).is_file()
    assert Path(report["markdown_path"]).is_file()


def test_projection_seed_sweep_dry_run(tmp_path):
    summary = run_projection_seed_sweep(
        base_config_path=ROOT / "configs" / "mujoco_full_car_safe_ppo.yaml",
        out_dir=tmp_path / "seed_sweep",
        seeds=[42, 43],
        episodes=1,
        expert_episodes=1,
        expert_max_steps=2,
        train_scenario_limit=1,
        eval_scenario_limit=1,
        episode_seconds=0.05,
        mujoco_settle_seconds=0.2,
        dry_run=True,
    )
    assert summary["dry_run"]
    assert summary["seed_count"] == 2
    assert summary["supported_seeds"] == 0
    assert Path(summary["json_path"]).is_file()
    assert Path(summary["csv_path"]).is_file()
    assert Path(summary["markdown_path"]).is_file()
    for seed in (42, 43):
        assert (tmp_path / "seed_sweep" / f"seed_{seed}" / "seed_config.yaml").is_file()
        assert (tmp_path / "seed_sweep" / f"seed_{seed}" / "ablation" / "si_rppo_ablation_manifest.json").is_file()


def test_preflight_mujoco_training_smoke(tmp_path):
    manifest = preflight_mujoco_training(
        config_path=ROOT / "configs" / "mujoco_full_car_corner.yaml",
        out_dir=tmp_path / "preflight",
        validation_steps=2,
        vector_envs=1,
        vector_steps=2,
        statistics_episodes=1,
        statistics_steps=2,
        action_mode="passive",
    )
    assert manifest["passed"]
    assert Path(manifest["manifest"]).is_file()
    for artifact in manifest["artifacts"].values():
        assert Path(artifact).is_file()


def test_mujoco_robustness_matrix_smoke(tmp_path):
    matrix = run_robustness_matrix(
        config_path=ROOT / "configs" / "mujoco_full_car_corner.yaml",
        out_dir=tmp_path / "matrix",
        max_steps=2,
        scenario_limit=1,
        max_unsafe_fraction=1.0,
    )
    assert matrix["passed"]
    assert len(matrix["cases"]) == 4
    assert Path(matrix["matrix_report"]).is_file()


def test_scenario_sampler_weighted_and_curriculum():
    scenarios = [{"name": "a"}, {"name": "b"}, {"name": "c"}]
    weighted = ScenarioSampler(
        scenarios,
        {"seed": 1, "scenario_sampling": {"mode": "weighted", "weights": {"a": 0.0, "b": 1.0, "c": 0.0}}},
    )
    assert [weighted.select(i)["name"] for i in range(5)] == ["b"] * 5
    curriculum = ScenarioSampler(
        scenarios,
        {
            "seed": 1,
            "scenario_sampling": {
                "mode": "cycle",
                "curriculum": [
                    {"until_episode": 2, "mode": "cycle", "scenarios": ["a"]},
                    {"until_episode": 4, "mode": "cycle", "scenarios": ["b"]},
                ],
            },
        },
    )
    assert [curriculum.select(i)["name"] for i in range(5)] == ["a", "a", "b", "b", "b"]


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
