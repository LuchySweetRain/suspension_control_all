from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import load_config
from roads.road_profiles import RoadProfileFactory


DEFAULT_DATASET_SCENARIOS = [
    {"name": "iso_b_22ms", "label": "ISO B 22 m/s", "type": "iso", "level": "B", "speed": 22.2222222222, "seed": 101},
    {"name": "iso_c_12ms", "label": "ISO C 12 m/s", "type": "iso", "level": "C", "speed": 12.0, "seed": 202},
    {"name": "sine_long_18ms", "label": "Long Sine 18 m/s", "type": "sine", "speed": 18.0, "amplitude": 0.025, "wavelength": 5.0},
    {"name": "bump_22ms", "label": "Bump 22 m/s", "type": "bump", "speed": 22.2222222222, "height": 0.08, "length": 0.35, "start_time": 0.8},
    {"name": "pothole_16ms", "label": "Pothole 16 m/s", "type": "pothole", "speed": 16.0, "depth": 0.06, "length": 0.55, "start_time": 0.9},
    {
        "name": "mixed_18ms",
        "label": "Mixed 18 m/s",
        "type": "composite",
        "speed": 18.0,
        "components": [
            {"type": "iso", "level": "B", "seed": 3021},
            {"type": "sine", "amplitude": 0.01, "wavelength": 3.5},
            {"type": "bump", "height": 0.04, "length": 0.45, "start_time": 1.6},
        ],
    },
]


def generate_dataset(
    out_dir: Path,
    duration: float = 8.0,
    dt: float = 0.001,
    base_config: Path | None = None,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    road_dir = out_dir / "roads"
    road_dir.mkdir(exist_ok=True)
    scenarios = []
    manifest = {"duration": duration, "dt": dt, "roads": []}
    for scenario in DEFAULT_DATASET_SCENARIOS:
        road = RoadProfileFactory.create(scenario, duration=duration, dt=dt)
        times = np.arange(0.0, duration + 0.5 * dt, dt)
        heights = np.asarray([road.value(float(t)) for t in times], dtype=np.float64)
        csv_path = road_dir / f"{scenario['name']}.csv"
        np.savetxt(csv_path, np.column_stack([times, heights]), delimiter=",", header="time,height", comments="")
        csv_scenario = {
            "name": scenario["name"],
            "label": scenario.get("label", scenario["name"]),
            "type": "csv",
            "path": str(csv_path),
            "speed": float(scenario.get("speed", 20.0)),
            "seed": int(scenario.get("seed", 42)),
        }
        scenarios.append(csv_scenario)
        manifest["roads"].append(
            {
                "name": scenario["name"],
                "source_type": scenario["type"],
                "csv": str(csv_path),
                "speed": csv_scenario["speed"],
                "min_height_m": float(np.min(heights)),
                "max_height_m": float(np.max(heights)),
                "rms_height_m": float(np.sqrt(np.mean(np.square(heights)))),
            }
        )
    with (out_dir / "road_dataset_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    if base_config is None:
        base_config = ROOT / "configs" / "mujoco_full_car_corner.yaml"
    config = load_config(base_config)
    config["scenarios"] = scenarios
    config.setdefault("evaluation", {})["controllers"] = ["PASSIVE"]
    config_path = out_dir / "mujoco_full_car_dataset.yaml"
    with config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
    manifest["config"] = str(config_path)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="datasets/mujoco_roads")
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--base-config", default=None)
    args = parser.parse_args()
    base_config = Path(args.base_config) if args.base_config else None
    if base_config is not None and not base_config.is_absolute():
        base_config = ROOT / base_config
    manifest = generate_dataset(Path(args.out), duration=args.duration, dt=args.dt, base_config=base_config)
    print(f"Generated {len(manifest['roads'])} roads under {Path(args.out).resolve()}")
    print(f"Evaluation config: {manifest['config']}")


if __name__ == "__main__":
    main()
