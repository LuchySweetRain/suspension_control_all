from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import load_config
from experiments.evaluation import evaluate_all
from scripts.generate_road_dataset import generate_dataset
from scripts.summarize_benchmark import build_report
from visualization.plots import plot_all


def _split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip().lower() for item in value.split(",") if item.strip()]


def _latest_checkpoint(algorithm: str, not_before: float = 0.0) -> tuple[Path, Path]:
    candidates = sorted(
        (ROOT / "results").glob(f"*_{algorithm}/checkpoints/final.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    candidates = [p for p in candidates if p.stat().st_mtime >= not_before]
    if not candidates:
        raise FileNotFoundError(f"No final checkpoint found for {algorithm} after benchmark training.")
    checkpoint = candidates[0]
    return checkpoint, checkpoint.parents[1]


def _train_algorithm(config_path: Path, algorithm: str, episodes: int) -> tuple[Path, Path]:
    before = datetime.now().timestamp() - 1.0
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "train_rl.py"),
        "--config",
        str(config_path),
        "--algorithm",
        algorithm,
        "--episodes",
        str(episodes),
    ]
    subprocess.run(cmd, cwd=ROOT, check=True)
    return _latest_checkpoint(algorithm, not_before=before)


def run_benchmark(
    config_path: Path,
    out_dir: Path,
    algorithms: list[str],
    episodes: int,
    generate_roads: bool,
    road_dataset_dir: Path | None,
    road_duration: float,
    road_dt: float,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    active_config_path = config_path
    road_manifest = None
    if generate_roads:
        road_dataset_dir = road_dataset_dir or out_dir / "road_dataset"
        road_manifest = generate_dataset(
            road_dataset_dir,
            duration=road_duration,
            dt=road_dt,
            base_config=config_path,
        )
        active_config_path = Path(road_manifest["config"])

    train_runs: dict[str, dict[str, str]] = {}
    checkpoints: dict[str, str] = {}
    if episodes > 0:
        for algorithm in algorithms:
            checkpoint, run_dir = _train_algorithm(active_config_path, algorithm, episodes)
            train_runs[algorithm] = {"run_dir": str(run_dir), "checkpoint": str(checkpoint)}
            checkpoints[algorithm] = str(checkpoint)

    config = load_config(active_config_path)
    eval_dir = out_dir / "evaluation"
    metrics, _ = evaluate_all(config, checkpoints=checkpoints, result_dir=eval_dir)
    plot_all(eval_dir)
    report_path = build_report(eval_dir)

    manifest = {
        "config": str(active_config_path),
        "base_config": str(config_path),
        "generated_roads": road_manifest,
        "algorithms": algorithms,
        "episodes": episodes,
        "train_runs": train_runs,
        "evaluation_dir": str(eval_dir),
        "benchmark_report": str(report_path),
        "metric_rows": int(len(metrics)),
    }
    with (out_dir / "benchmark_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mujoco_full_car_corner.yaml")
    parser.add_argument("--out", default=None)
    parser.add_argument("--algorithms", default="", help="Comma-separated RL algorithms to train/evaluate.")
    parser.add_argument("--episodes", type=int, default=0, help="Training episodes per algorithm. Use 0 for passive-only evaluation.")
    parser.add_argument("--generate-roads", action="store_true")
    parser.add_argument("--road-dataset-dir", default=None)
    parser.add_argument("--road-duration", type=float, default=8.0)
    parser.add_argument("--road-dt", type=float, default=0.001)
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    out_dir = Path(args.out) if args.out else ROOT / "results" / datetime.now().strftime("%Y%m%d_%H%M%S_mujoco_benchmark")
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    road_dataset_dir = Path(args.road_dataset_dir) if args.road_dataset_dir else None
    if road_dataset_dir is not None and not road_dataset_dir.is_absolute():
        road_dataset_dir = ROOT / road_dataset_dir
    manifest = run_benchmark(
        config_path=config_path,
        out_dir=out_dir,
        algorithms=_split_csv(args.algorithms),
        episodes=args.episodes,
        generate_roads=args.generate_roads,
        road_dataset_dir=road_dataset_dir,
        road_duration=args.road_duration,
        road_dt=args.road_dt,
    )
    print(f"Saved benchmark to {out_dir}")
    print(f"Report: {manifest['benchmark_report']}")


if __name__ == "__main__":
    main()
