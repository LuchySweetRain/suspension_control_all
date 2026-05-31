from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import load_config


def _safe_name(path: Path) -> str:
    return re.sub(r"[^0-9A-Za-z_]+", "_", path.stem).strip("_").lower() or "road"


def _read_road_stats(path: Path, delimiter: str, skiprows: int, scale: float) -> dict:
    try:
        data = np.loadtxt(path, delimiter=delimiter, skiprows=skiprows)
    except ValueError:
        data = np.loadtxt(path, delimiter=delimiter, skiprows=skiprows + 1)
    if data.ndim == 1:
        data = data.reshape(-1, 2)
    if data.shape[1] < 2:
        raise ValueError(f"{path} must contain at least two columns: time,height")
    times = np.asarray(data[:, 0], dtype=np.float64)
    heights = np.asarray(data[:, 1], dtype=np.float64) * scale
    duration = float(np.max(times) - np.min(times)) if len(times) else 0.0
    dt = float(np.median(np.diff(times))) if len(times) > 1 else 0.0
    return {
        "samples": int(len(heights)),
        "duration_s": duration,
        "dt_s": dt,
        "min_height_m": float(np.min(heights)),
        "max_height_m": float(np.max(heights)),
        "rms_height_m": float(np.sqrt(np.mean(np.square(heights)))),
    }


def import_road_directory(
    road_dir: Path,
    out_dir: Path,
    base_config: Path | None = None,
    pattern: str = "*.csv",
    speed: float = 20.0,
    delimiter: str = ",",
    skiprows: int = 0,
    scale: float = 1.0,
) -> dict:
    files = sorted(road_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No road files matched {pattern!r} under {road_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    if base_config is None:
        base_config = ROOT / "configs" / "mujoco_full_car_corner.yaml"
    config = load_config(base_config)

    scenarios = []
    manifest = {
        "source_dir": str(road_dir.resolve()),
        "pattern": pattern,
        "base_config": str(base_config.resolve()),
        "speed": float(speed),
        "delimiter": delimiter,
        "skiprows": int(skiprows),
        "scale": float(scale),
        "roads": [],
    }

    used_names: set[str] = set()
    for index, path in enumerate(files):
        name = _safe_name(path)
        if name in used_names:
            name = f"{name}_{index}"
        used_names.add(name)
        stats = _read_road_stats(path, delimiter=delimiter, skiprows=skiprows, scale=scale)
        scenario = {
            "name": name,
            "label": path.stem,
            "type": "csv",
            "path": str(path.resolve()),
            "speed": float(speed),
            "delimiter": delimiter,
            "skiprows": int(skiprows),
            "scale": float(scale),
        }
        scenarios.append(scenario)
        manifest["roads"].append({"name": name, "path": str(path.resolve()), **stats})

    config["scenarios"] = scenarios
    config.setdefault("evaluation", {})["controllers"] = ["PASSIVE"]
    config_path = out_dir / "mujoco_full_car_imported_roads.yaml"
    manifest_path = out_dir / "imported_road_manifest.json"
    with config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
    manifest["config"] = str(config_path.resolve())
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    manifest["manifest"] = str(manifest_path.resolve())
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--road-dir", required=True)
    parser.add_argument("--out", default="datasets/imported_roads")
    parser.add_argument("--base-config", default=None)
    parser.add_argument("--pattern", default="*.csv")
    parser.add_argument("--speed", type=float, default=20.0)
    parser.add_argument("--delimiter", default=",")
    parser.add_argument("--skiprows", type=int, default=0)
    parser.add_argument("--scale", type=float, default=1.0)
    args = parser.parse_args()
    base_config = Path(args.base_config) if args.base_config else None
    if base_config is not None and not base_config.is_absolute():
        base_config = ROOT / base_config
    manifest = import_road_directory(
        road_dir=Path(args.road_dir),
        out_dir=Path(args.out),
        base_config=base_config,
        pattern=args.pattern,
        speed=args.speed,
        delimiter=args.delimiter,
        skiprows=args.skiprows,
        scale=args.scale,
    )
    print(f"Imported {len(manifest['roads'])} roads from {Path(args.road_dir).resolve()}")
    print(f"Evaluation config: {manifest['config']}")


if __name__ == "__main__":
    main()
