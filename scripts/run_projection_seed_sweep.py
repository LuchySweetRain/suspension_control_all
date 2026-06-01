from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import load_config
from scripts.run_si_rppo_ablation import run_si_rppo_ablation


CORE_METRICS = [
    "EpisodeReturn",
    "UnsafeSteps",
    "BodyAccRMS_mps2",
    "PitchAccRMS_radps2",
    "RollAccRMS_radps2",
    "ActionDeltaRMS_N",
    "ActuatorTrackingRMS_N",
    "PolicyProjectionError",
]


def _write_seed_config(base_config_path: Path, seed: int, out_path: Path) -> Path:
    config = load_config(base_config_path)
    config["seed"] = int(seed)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return out_path


def _load_claim_report(run_dir: Path) -> dict:
    report_path = run_dir / "si_rppo_claim_report.json"
    if not report_path.is_file():
        return {"status": "missing_report", "core_status": "missing_report", "comparisons": []}
    return json.loads(report_path.read_text(encoding="utf-8"))


def _summarize_seed_metrics(run_dir: Path, seed: int) -> dict:
    combined_path = run_dir / "combined_metrics.csv"
    row: dict[str, float | str | int | None] = {"seed": int(seed)}
    if not combined_path.is_file():
        row["metrics_status"] = "missing_metrics"
        return row
    try:
        metrics = pd.read_csv(combined_path)
    except EmptyDataError:
        row["metrics_status"] = "empty_metrics"
        return row
    if metrics.empty or "Variant" not in metrics.columns or "Controller" not in metrics.columns:
        row["metrics_status"] = "empty_metrics"
        return row
    grouped = metrics.groupby(["Variant", "Controller"], dropna=False).mean(numeric_only=True)
    for variant in ("ppo_scratch", "bc_ppo"):
        key = (variant, "PPO")
        if key not in grouped.index:
            row[f"{variant}_status"] = "missing"
            continue
        row[f"{variant}_status"] = "ready"
        for metric in CORE_METRICS:
            if metric in grouped.columns and pd.notna(grouped.loc[key, metric]):
                row[f"{variant}_{metric}"] = float(grouped.loc[key, metric])
    for metric in CORE_METRICS:
        candidate_key = f"bc_ppo_{metric}"
        baseline_key = f"ppo_scratch_{metric}"
        if candidate_key in row and baseline_key in row:
            row[f"delta_{metric}"] = float(row[candidate_key] - row[baseline_key])  # type: ignore[operator]
    row["metrics_status"] = "ready"
    return row


def _build_sweep_report(summary: dict, out_dir: Path) -> None:
    md_path = out_dir / "projection_seed_sweep_report.md"
    rows = summary.get("seeds", [])
    lines = [
        "# Projection-Aware PPO Seed Sweep",
        "",
        f"- Status: `{summary['status']}`",
        f"- Supported seeds: {summary['supported_seeds']} / {summary['seed_count']}",
        f"- Support rate: {summary['support_rate']:.3f}",
        "",
        "## Per-Seed Core Claim",
        "",
        "| Seed | CoreStatus | ReturnDelta | UnsafeDelta | ActionDeltaDelta | ProjectionErrorDelta |",
        "| ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["seed"]),
                    str(row.get("core_status", "")),
                    "" if row.get("delta_EpisodeReturn") is None else f"{row['delta_EpisodeReturn']:.4f}",
                    "" if row.get("delta_UnsafeSteps") is None else f"{row['delta_UnsafeSteps']:.4f}",
                    "" if row.get("delta_ActionDeltaRMS_N") is None else f"{row['delta_ActionDeltaRMS_N']:.4f}",
                    "" if row.get("delta_PolicyProjectionError") is None else f"{row['delta_PolicyProjectionError']:.4f}",
                ]
            )
            + " |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- `ReturnDelta` is BC-PPO minus PPO-scratch, so higher is better.",
        "- Safety, smoothness, actuator, and projection deltas are BC-PPO minus PPO-scratch, so lower is better.",
        "- Treat the core claim as robust only when all planned seeds report `supported` or when unsupported seeds have a documented, reproducible cause.",
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")


def run_projection_seed_sweep(
    base_config_path: Path,
    out_dir: Path,
    seeds: list[int],
    episodes: int,
    expert_episodes: int,
    expert_max_steps: int | None = None,
    train_scenario_limit: int | None = None,
    eval_scenario_limit: int | None = None,
    episode_seconds: float | None = None,
    mujoco_settle_seconds: float | None = None,
    dry_run: bool = False,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_rows = []
    for seed in seeds:
        seed_dir = out_dir / f"seed_{seed}"
        seed_config = _write_seed_config(base_config_path, int(seed), seed_dir / "seed_config.yaml")
        manifest = run_si_rppo_ablation(
            base_config_path=seed_config,
            out_dir=seed_dir / "ablation",
            episodes=episodes,
            expert_episodes=expert_episodes,
            expert_max_steps=expert_max_steps,
            expert_controller="PASSIVE",
            skip_unsafe_expert=True,
            train_scenario_limit=train_scenario_limit,
            eval_scenario_limit=eval_scenario_limit,
            episode_seconds=episode_seconds,
            mujoco_settle_seconds=mujoco_settle_seconds,
            variants=["ppo_scratch", "bc_ppo"],
            baseline_algorithms=None,
            dry_run=dry_run,
        )
        claim_report = _load_claim_report(Path(manifest["out_dir"]))
        row = _summarize_seed_metrics(Path(manifest["out_dir"]), int(seed))
        row.update(
            {
                "seed": int(seed),
                "manifest": str((Path(manifest["out_dir"]) / "si_rppo_ablation_manifest.json").resolve()),
                "claim_report": str(Path(claim_report.get("json_path", Path(manifest["out_dir"]) / "si_rppo_claim_report.json")).resolve())
                if claim_report.get("json_path")
                else str((Path(manifest["out_dir"]) / "si_rppo_claim_report.json").resolve()),
                "status": claim_report.get("status", "missing_report"),
                "core_status": claim_report.get("core_status", "missing_report"),
            }
        )
        seed_rows.append(row)

    supported = sum(1 for row in seed_rows if row.get("core_status") == "supported")
    summary = {
        "base_config": str(base_config_path.resolve()),
        "out_dir": str(out_dir.resolve()),
        "episodes": int(episodes),
        "expert_episodes": int(expert_episodes),
        "expert_max_steps": expert_max_steps,
        "dry_run": bool(dry_run),
        "seeds": seed_rows,
        "seed_count": len(seed_rows),
        "supported_seeds": int(supported),
        "support_rate": float(supported / max(1, len(seed_rows))),
        "status": "supported" if supported == len(seed_rows) and seed_rows else "needs_more_evidence",
    }
    json_path = out_dir / "projection_seed_sweep_summary.json"
    csv_path = out_dir / "projection_seed_sweep_summary.csv"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(seed_rows).to_csv(csv_path, index=False)
    _build_sweep_report(summary, out_dir)
    summary["json_path"] = str(json_path.resolve())
    summary["csv_path"] = str(csv_path.resolve())
    summary["markdown_path"] = str((out_dir / "projection_seed_sweep_report.md").resolve())
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mujoco_full_car_safe_ppo.yaml")
    parser.add_argument("--out", default=None)
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--expert-episodes", type=int, default=20)
    parser.add_argument("--expert-max-steps", type=int, default=None)
    parser.add_argument("--train-scenario-limit", type=int, default=None)
    parser.add_argument("--eval-scenario-limit", type=int, default=None)
    parser.add_argument("--episode-seconds", type=float, default=None)
    parser.add_argument("--mujoco-settle-seconds", type=float, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    out_dir = Path(args.out) if args.out else ROOT / "results" / datetime.now().strftime("%Y%m%d_%H%M%S_projection_seed_sweep")
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    summary = run_projection_seed_sweep(
        base_config_path=config_path,
        out_dir=out_dir,
        seeds=seeds,
        episodes=args.episodes,
        expert_episodes=args.expert_episodes,
        expert_max_steps=args.expert_max_steps,
        train_scenario_limit=args.train_scenario_limit,
        eval_scenario_limit=args.eval_scenario_limit,
        episode_seconds=args.episode_seconds,
        mujoco_settle_seconds=args.mujoco_settle_seconds,
        dry_run=args.dry_run,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
