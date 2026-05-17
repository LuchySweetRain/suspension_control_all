from __future__ import annotations

from pathlib import Path
from copy import deepcopy

import numpy as np
import pandas as pd

from controllers import MPCController, PIDController, RLTransformerTD3Controller, SPDFController
from envs import HalfCarEnv


def make_controller(name: str, env: HalfCarEnv, config: dict, checkpoint: str | None = None):
    key = name.lower()
    if key == "pid":
        return PIDController(env.params, env.control_dt, env.force_limit)
    if key == "spdf":
        return SPDFController(env.params, env.control_dt)
    if key == "mpc":
        return MPCController(env.model, config)
    if key == "rl":
        return RLTransformerTD3Controller(config, checkpoint=checkpoint)
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
        records.append(
            {
                "time": info["time"],
                "zb": state[0],
                "dzb": state[1],
                "theta": state[2],
                "dtheta": state[3],
                "zwf": state[4],
                "zwr": state[6],
                "ddzb": d["ddzb"],
                "ddtheta": d["ddtheta"],
                "delta_yf": d["delta_yf"],
                "delta_yr": d["delta_yr"],
                "Fpf": d["Fpf"],
                "Fpr": d["Fpr"],
                "Uaf": action[0],
                "Uar": action[1],
                "reward": reward,
                "unsafe": int(info["unsafe"]),
                "preview_used": int(use_preview),
            }
        )
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
    return {
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


def evaluate_all(config: dict, checkpoint: str | None, result_dir: Path) -> tuple[pd.DataFrame, dict]:
    result_dir.mkdir(parents=True, exist_ok=True)
    controllers = ["PID", "SPDF", "MPC", "RL"] if checkpoint else ["PID", "SPDF", "MPC"]
    trajectories = {}
    rows = []
    for scenario in config["scenarios"]:
        for controller_name in controllers:
            use_preview = controller_name.lower() in set(config["preview"]["enabled_for"])
            controller_config = deepcopy(config)
            if controller_name.lower() in {"pid", "spdf"}:
                # Match active_suspension_sim.slx: classical controllers update at
                # the fixed solver step, while MPC/RL use the preview/control step.
                controller_config["control_dt"] = controller_config["dt"]
            env = HalfCarEnv(controller_config, scenario=scenario, use_preview=use_preview)
            controller = make_controller(controller_name, env, controller_config, checkpoint=checkpoint)
            run = rollout(env, controller, use_preview=use_preview)
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
                    **metrics,
                }
            )
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(result_dir / "metrics.csv", index=False)
    return metrics_df, trajectories
