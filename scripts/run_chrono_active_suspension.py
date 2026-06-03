"""Run a passive-vs-active quarter-car smoke benchmark with optional PyChrono."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.chrono_quarter_car_env import ChronoQuarterCarEnv, QuarterCarParams


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def write_rollout(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/chrono_quarter_car.yaml"))
    parser.add_argument("--out", type=Path, default=Path("results/chrono_quarter_car_smoke"))
    parser.add_argument("--backend", choices=["rk4", "chrono"], default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    backend = args.backend or str(cfg.get("backend", "rk4"))
    params = QuarterCarParams(**(cfg.get("params") or {}))
    env = ChronoQuarterCarEnv(params=params, dt=float(cfg.get("dt", 0.001)), backend=backend)
    duration = float(cfg.get("duration", 2.0))

    manifest = {
        "config": str(args.config.resolve()),
        "backend": backend,
        "duration": duration,
        "dt": float(cfg.get("dt", 0.001)),
        "controllers": {},
    }
    for controller in ["passive", "active"]:
        result = env.rollout(duration=duration, controller=controller)
        write_rollout(args.out / f"{controller}_rollout.csv", result["rows"])
        manifest["controllers"][controller] = result["metrics"]

    passive = manifest["controllers"]["passive"]
    active = manifest["controllers"]["active"]
    manifest["improvements"] = {
        key: active[key] - passive[key]
        for key in ["BodyAccRMS_mps2", "BodyDispRMS_m", "MaxSuspensionDeflection_m", "UnsafeSteps"]
    }
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
