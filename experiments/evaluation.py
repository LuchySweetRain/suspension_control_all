from __future__ import annotations

from pathlib import Path
from copy import deepcopy
import json

import numpy as np
import pandas as pd

from controllers import MPCController, PIDController, ReducedFullCarPreviewController, RLTransformerController, SPDFController
from controllers.prior import adapt_action_to_env, compute_prior_action, make_prior_controller
from envs import HalfCarEnv, MuJoCoFullCarEnv, MuJoCoVehicleEnv


class PassiveController:
    def __init__(self, act_dim: int):
        self.act_dim = int(act_dim)

    def reset(self):
        pass

    def compute_action(self, obs, info):
        return np.zeros(self.act_dim, dtype=np.float32)


class RandomController:
    def __init__(self, env, seed: int = 42):
        self.env = env
        self.rng = np.random.default_rng(seed)

    def reset(self):
        pass

    def compute_action(self, obs, info):
        if hasattr(self.env.action_space, "low") and hasattr(self.env.action_space, "high"):
            return self.rng.uniform(self.env.action_space.low, self.env.action_space.high).astype(np.float32)
        return self.env.action_space.sample()


class ResidualController:
    def __init__(self, prior, residual, env, residual_scale: float):
        self.prior = prior
        self.residual = residual
        self.env = env
        self.residual_scale = float(residual_scale)

    def reset(self):
        if self.prior is not None:
            self.prior.reset()
        self.residual.reset()

    def compute_action(self, obs, info):
        prior_action = compute_prior_action(self.prior, self.env, obs, info)
        residual_action = adapt_action_to_env(self.residual.compute_action(obs, info), self.env)
        return np.clip(
            prior_action + self.residual_scale * residual_action,
            self.env.action_space.low,
            self.env.action_space.high,
        ).astype(np.float32)


def make_env(config: dict, scenario: dict, use_preview: bool = True):
    engine = str(config.get("environment", {}).get("engine", "python")).lower()
    if engine in {"mujoco_full", "mujoco_full_car", "full_car"}:
        return MuJoCoFullCarEnv(config, scenario=scenario, use_preview=use_preview)
    if engine in {"mujoco", "mujoco_vehicle"}:
        return MuJoCoVehicleEnv(config, scenario=scenario, use_preview=use_preview)
    return HalfCarEnv(config, scenario=scenario, use_preview=use_preview)


def make_controller(
    name: str,
    env: HalfCarEnv,
    config: dict,
    checkpoint: str | None = None,
    algorithm: str | None = None,
):
    key = name.lower()
    if key in {"passive", "zero", "none"}:
        return PassiveController(env.action_space.shape[0] if hasattr(env, "action_space") else 2)
    if key == "random":
        return RandomController(env, int(config.get("seed", 42)))
    if key == "pid":
        return PIDController(env.params, env.control_dt, env.force_limit)
    if key == "spdf":
        return SPDFController(env.params, env.control_dt)
    if key == "mpc":
        if not hasattr(env, "model"):
            raise ValueError("MPC evaluation requires the Python HalfCarEnv model.")
        return MPCController(env.model, config)
    if key in {"full_car_mpc_lite", "mpc_lite", "lqr", "lpv"}:
        return ReducedFullCarPreviewController(env, config)
    if key in {"td3", "ddpg", "sac", "ppo", "rl"}:
        rl = RLTransformerController(config, algorithm=algorithm or key, checkpoint=checkpoint)
        residual_cfg = dict(config.get("residual_control", {}))
        if residual_cfg.get("enabled", False):
            prior = make_prior_controller(str(residual_cfg.get("prior", "spdf")), env, config)
            return ResidualController(prior, rl, env, float(residual_cfg.get("scale", 1.0)))
        return rl
    raise ValueError(f"Unknown controller: {name}")


def rollout(env: HalfCarEnv, controller, use_preview: bool) -> dict:
    obs, info = env.reset(options={"scenario": env.scenario})
    controller.reset()
    records = []
    done = False
    while not done:
        obs_for_controller = obs if use_preview else obs[: env.base_obs_dim]
        action = controller.compute_action(obs_for_controller, info)
        obs, reward, terminated, truncated, info = env.step(action)
        d = info["derived"]
        state = info["state"]
        road = info["road"]
        row = {
                "time": info["time"],
                "zb": state[0],
                "dzb": state[1],
                "theta": state[2],
                "dtheta": state[3],
                "zwf": state[4],
                "zwr": state[6],
                "zdf": road[0],
                "zdr": road[2],
                "ddzb": d["ddzb"],
                "ddtheta": d["ddtheta"],
                "delta_yf": d["delta_yf"],
                "delta_yr": d["delta_yr"],
                "Fpf": d["Fpf"],
                "Fpr": d["Fpr"],
                "Uaf": d.get("Uaf", action[0]),
                "Uar": d.get("Uar", action[1] if len(action) > 1 else 0.0),
                "reward": reward,
                "unsafe": int(info["unsafe"]),
                "preview_used": int(use_preview),
            }
        if len(state) >= 14:
            row.update(
                {
                    "roll": state[4],
                    "droll": state[5],
                    "zwfl": state[6],
                    "zwfr": state[8],
                    "zwrl": state[10],
                    "zwrr": state[12],
                }
            )
        if "road_corners" in info:
            rc = info["road_corners"]
            row.update(
                {
                    "zdfl": rc[0],
                    "zdfr": rc[2],
                    "zdrl": rc[4],
                    "zdrr": rc[6],
                }
            )
        if "corner_action" in info:
            ca = info["corner_action"]
            row.update({"Ufl": ca[0], "Ufr": ca[1], "Url": ca[2], "Urr": ca[3]})
        if "command_corner_action" in info:
            cc = info["command_corner_action"]
            row.update({"Ucmd_fl": cc[0], "Ucmd_fr": cc[1], "Ucmd_rl": cc[2], "Ucmd_rr": cc[3]})
        if "command_action" in info:
            cmd = info["command_action"]
            row["Ucmd_af"] = cmd[0]
            row["Ucmd_ar"] = cmd[1] if len(cmd) > 1 else 0.0
        for metric_name, value in info.get("action_metrics", {}).items():
            if np.isscalar(value):
                row[f"action_{metric_name}"] = float(value)
        for component_name, value in info.get("reward_components", {}).items():
            row[f"reward_{component_name}"] = float(value)
        if "ddroll" in d:
            row["ddroll"] = d["ddroll"]
        for corner in ("fl", "fr", "rl", "rr"):
            if f"delta_y_{corner}" in d:
                row[f"delta_y_{corner}"] = d[f"delta_y_{corner}"]
                row[f"F_tire_{corner}"] = d[f"F_tire_{corner}"]
        records.append(row)
        done = terminated or truncated
    return {"data": pd.DataFrame(records), "return": float(sum(r["reward"] for r in records))}


def compute_metrics(df: pd.DataFrame, settle_seconds: float = 1.0) -> dict:
    view = df[df["time"] >= settle_seconds]
    if view.empty:
        view = df

    def rms(name: str) -> float:
        return float(np.sqrt(np.mean(np.square(view[name].to_numpy()))))

    def peak(name: str) -> float:
        return float(np.max(np.abs(view[name].to_numpy())))

    dt = float(np.mean(np.diff(df["time"].to_numpy()))) if len(df) > 1 else 0.01
    power = df["Uaf"].to_numpy() * df["delta_yf"].diff().fillna(0.0).to_numpy() / dt
    power += df["Uar"].to_numpy() * df["delta_yr"].diff().fillna(0.0).to_numpy() / dt
    metrics = {
        "BodyAccRMS_mps2": rms("ddzb"),
        "BodyAccPeak_mps2": peak("ddzb"),
        "PitchAccRMS_radps2": rms("ddtheta"),
        "PitchAccPeak_radps2": peak("ddtheta"),
        "MaxPitch_rad": peak("theta"),
        "MaxFrontSusp_m": peak("delta_yf"),
        "MaxRearSusp_m": peak("delta_yr"),
        "FrontTireLoadRMS_N": rms("Fpf"),
        "RearTireLoadRMS_N": rms("Fpr"),
        "MaxUaf_N": peak("Uaf"),
        "MaxUar_N": peak("Uar"),
        "ControlAbsEnergy_J": float(np.sum(np.abs(power)) * dt),
        "UnsafeSteps": int(view["unsafe"].sum()),
        "EpisodeReturn": float(df["reward"].sum()),
    }
    for column in (
        "action_action_delta_rms",
        "action_command_delta_rms",
        "action_actuator_tracking_rms",
        "action_saturation_ratio",
    ):
        if column in df.columns:
            label = {
                "action_action_delta_rms": "ActionDeltaRMS_N",
                "action_command_delta_rms": "CommandDeltaRMS_N",
                "action_actuator_tracking_rms": "ActuatorTrackingRMS_N",
                "action_saturation_ratio": "ActuatorSaturationRatio",
            }[column]
            metrics[label] = float(view[column].mean())
    for column in [name for name in df.columns if name.startswith("reward_")]:
        metrics[f"Mean{column.removeprefix('reward_').title().replace('_', '')}Penalty"] = float(view[column].mean())
    if "ddroll" in df.columns:
        metrics.update(
            {
                "RollAccRMS_radps2": rms("ddroll"),
                "RollAccPeak_radps2": peak("ddroll"),
                "MaxRoll_rad": peak("roll") if "roll" in df.columns else 0.0,
            }
        )
    for corner in ("fl", "fr", "rl", "rr"):
        susp_name = f"delta_y_{corner}"
        tire_name = f"F_tire_{corner}"
        action_name = f"U{corner}"
        if susp_name in df.columns:
            metrics[f"MaxSusp_{corner}_m"] = peak(susp_name)
        if tire_name in df.columns:
            metrics[f"TireLoadRMS_{corner}_N"] = rms(tire_name)
        if action_name in df.columns:
            metrics[f"MaxU_{corner}_N"] = peak(action_name)
        command_name = f"Ucmd_{corner}"
        if action_name in df.columns and command_name in df.columns:
            metrics[f"ActuatorTrackRMS_{corner}_N"] = float(
                np.sqrt(np.mean(np.square((view[command_name] - view[action_name]).to_numpy())))
            )
    return metrics


def evaluate_all(config: dict, checkpoints: str | dict[str, str] | None, result_dir: Path) -> tuple[pd.DataFrame, dict]:
    result_dir.mkdir(parents=True, exist_ok=True)
    with (result_dir / "evaluation_config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    if isinstance(checkpoints, str):
        rl_checkpoints = {"td3": checkpoints}
    else:
        rl_checkpoints = checkpoints or {}
    controllers = list(config.get("evaluation", {}).get("controllers", []))
    if not controllers:
        engine = str(config.get("environment", {}).get("engine", "python")).lower()
        controllers = ["PASSIVE"] if engine != "python" else ["PID", "SPDF", "MPC"]
    controllers += [name.upper() for name in rl_checkpoints if name.upper() not in controllers]
    trajectories = {}
    environment_metadata = []
    rows = []
    for scenario in config["scenarios"]:
        for controller_name in controllers:
            is_rl = controller_name.lower() in rl_checkpoints
            use_preview = is_rl or controller_name.lower() in set(config["preview"]["enabled_for"])
            controller_config = deepcopy(config)
            if controller_name.lower() in {"pid", "spdf"}:
                # Match active_suspension_sim.slx: classical controllers update at
                # the fixed solver step, while MPC/RL use the preview/control step.
                controller_config["control_dt"] = controller_config["dt"]
            env = make_env(controller_config, scenario=scenario, use_preview=use_preview)
            algorithm = controller_name.lower() if is_rl else None
            checkpoint = rl_checkpoints.get(algorithm) if algorithm else None
            controller = make_controller(
                controller_name,
                env,
                controller_config,
                checkpoint=checkpoint,
                algorithm=algorithm,
            )
            run = rollout(env, controller, use_preview=use_preview)
            if hasattr(env, "environment_metadata"):
                environment_metadata.append(
                    {
                        "controller": controller_name,
                        "scenario": scenario.get("name", ""),
                        **env.environment_metadata(),
                    }
                )
            if hasattr(env, "close"):
                env.close()
            df = run["data"]
            key = f"{controller_name}_{scenario['name']}"
            trajectories[key] = df
            df.to_csv(result_dir / f"{key}.csv", index=False)
            metrics = compute_metrics(df, float(config.get("settle_seconds", 1.0)))
            rows.append(
                {
                    "Controller": controller_name,
                    "Scenario": scenario["name"],
                    "ScenarioLabel": scenario.get("label", scenario["name"]),
                    "PreviewUsed": use_preview,
                    "Engine": str(config.get("environment", {}).get("engine", "python")),
                    **metrics,
                }
            )
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(result_dir / "metrics.csv", index=False)
    manifest = {
        "engine": str(config.get("environment", {}).get("engine", "python")),
        "controllers": controllers,
        "scenarios": [s.get("name", "") for s in config["scenarios"]],
        "checkpoints": rl_checkpoints,
        "trajectory_files": sorted(f"{key}.csv" for key in trajectories),
        "metric_columns": list(metrics_df.columns),
        "environment_metadata": environment_metadata,
    }
    with (result_dir / "evaluation_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return metrics_df, trajectories
