from __future__ import annotations

import json
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
from envs import HalfCarEnv, MuJoCoFullCarEnv, MuJoCoHalfCarEnv, MuJoCoVehicleEnv
from experiments.evaluation import evaluate_all, make_controller
from scripts.benchmark_vector_env import benchmark_vector_env
from scripts.export_mujoco_env_spec import export_env_spec
from scripts.generate_road_dataset import generate_dataset
from scripts.import_road_directory import import_road_directory
from scripts.preflight_mujoco_training import preflight_mujoco_training
from scripts.check_gym_registration import check_registration
from scripts.collect_env_statistics import collect_env_statistics
from scripts.run_mujoco_benchmark import run_benchmark
from scripts.run_mujoco_robustness_matrix import run_robustness_matrix
from scripts.summarize_benchmark import build_report
from scripts.train_rl import ScenarioSampler
from scripts.validate_training_config import validate_config
from scripts.validate_mujoco_env import validate_environment
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
