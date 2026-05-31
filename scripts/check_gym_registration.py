from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs import GYM_IDS, register_gymnasium_envs


def check_registration(config_path: Path | None = None, smoke: bool = False) -> dict:
    registered = register_gymnasium_envs()
    report = {"gymnasium_available": registered, "ids": dict(GYM_IDS), "smoke": None}
    if not registered:
        return report
    if smoke:
        import gymnasium as gym

        kwargs = {"config_path": str(config_path)} if config_path else {}
        env = gym.make(GYM_IDS["mujoco_full_car"], scenario_index=0, use_preview=True, width=160, height=90, **kwargs)
        try:
            obs, info = env.reset(seed=42)
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, step_info = env.step(action)
            report["smoke"] = {
                "obs_shape": list(obs.shape),
                "next_obs_shape": list(next_obs.shape),
                "action_shape": list(env.action_space.shape),
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "engine": step_info.get("engine", info.get("engine", "")),
            }
        finally:
            env.close()
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mujoco_full_car_corner.yaml")
    parser.add_argument("--out", default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    report = check_registration(config_path=config_path, smoke=args.smoke)
    text = json.dumps(report, indent=2)
    if args.out:
        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_path = ROOT / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")
    print(text)
    if not report["gymnasium_available"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
