from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import load_config
from experiments.evaluation import evaluate_all
from visualization.plots import plot_all


def resolve_checkpoint(value: str | None, algorithm: str = "td3") -> str | None:
    if not value:
        return None
    if "<run>" in value:
        raise ValueError(
            "Checkpoint path still contains the placeholder '<run>'. "
            "Use a real run directory, for example "
            "results\\20260517_180625_td3\\checkpoints\\final.pt."
        )
    if value.lower() == "latest":
        candidates = sorted(
            (ROOT / "results").glob(f"*_{algorithm}/checkpoints/final.pt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise FileNotFoundError(
                f"No {algorithm.upper()} checkpoint found under results\\*_{algorithm}\\checkpoints\\final.pt"
            )
        return str(candidates[0])
    checkpoint = Path(value)
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    return str(checkpoint)


def resolve_checkpoints(value: str | None) -> dict[str, str]:
    if not value:
        return {}
    resolved = {}
    for item in value.split(","):
        if not item.strip():
            continue
        if "=" not in item:
            raise ValueError("Use --checkpoints as algorithm=path pairs, for example td3=latest,sac=latest")
        algorithm, checkpoint = item.split("=", 1)
        algorithm = algorithm.strip().lower()
        resolved[algorithm] = resolve_checkpoint(checkpoint.strip(), algorithm=algorithm)
    return resolved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--checkpoints",
        default=None,
        help="Comma-separated algorithm=checkpoint pairs, e.g. td3=latest,ddpg=latest,sac=latest,ppo=latest",
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    config = load_config(ROOT / args.config)
    checkpoints = resolve_checkpoints(args.checkpoints)
    result_dir = Path(args.out) if args.out else ROOT / "results" / datetime.now().strftime("%Y%m%d_%H%M%S_eval")
    metrics, _ = evaluate_all(config, checkpoints, result_dir)
    print(metrics)
    plot_all(result_dir)
    print(f"Saved evaluation to {result_dir}")


if __name__ == "__main__":
    main()
