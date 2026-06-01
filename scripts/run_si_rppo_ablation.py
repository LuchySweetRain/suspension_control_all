from __future__ import annotations

import argparse
import json
import subprocess
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import deep_update, load_config
from experiments.evaluation import evaluate_all
from scripts.collect_expert_dataset import collect_expert_dataset
from scripts.summarize_benchmark import build_report


VARIANTS = {
    "ppo_scratch": {
        "residual_control": {"enabled": False},
        "imitation": {"enabled": False},
        "evaluation": {"controllers": ["PASSIVE", "FULL_CAR_MPC_LITE"]},
        "rl": {"ppo": {"projection_penalty_weight": 0.0}},
        "reward": {"deviation": 0.0},
    },
    "bc_ppo": {
        "residual_control": {"enabled": False},
        "imitation": {"enabled": True, "residual_targets": False, "anchor_enabled": True},
        "evaluation": {"controllers": ["PASSIVE", "FULL_CAR_MPC_LITE"]},
        "reward": {"deviation": 0.0},
    },
    "residual_bc_ppo": {
        "residual_control": {"enabled": True, "gate": {"enabled": False}, "shield": {"enabled": False}},
        "imitation": {"enabled": True, "residual_targets": True, "anchor_enabled": False},
        "evaluation": {"controllers": ["PASSIVE", "FULL_CAR_MPC_LITE"]},
        "reward": {"deviation": 0.0},
    },
    "safe_residual_bc_ppo": {
        "residual_control": {"enabled": True, "gate": {"enabled": True}},
        "imitation": {"enabled": True, "residual_targets": True, "anchor_enabled": False},
        "evaluation": {"controllers": ["PASSIVE", "FULL_CAR_MPC_LITE"]},
    },
}

CLAIM_METRICS = [
    "EpisodeReturn",
    "UnsafeSteps",
    "BodyAccRMS_mps2",
    "PitchAccRMS_radps2",
    "RollAccRMS_radps2",
    "ActionDeltaRMS_N",
    "CommandDeltaRMS_N",
    "ActuatorTrackingRMS_N",
    "ActuatorSaturationRatio",
    "ActionDeviationRMS_N",
    "PolicyProjectionError",
    "PolicyProjectionDeltaRMS_N",
]

LOWER_IS_BETTER = {
    "UnsafeSteps",
    "BodyAccRMS_mps2",
    "PitchAccRMS_radps2",
    "RollAccRMS_radps2",
    "ActionDeltaRMS_N",
    "CommandDeltaRMS_N",
    "ActuatorTrackingRMS_N",
    "ActuatorSaturationRatio",
    "ActionDeviationRMS_N",
    "PolicyProjectionError",
    "PolicyProjectionDeltaRMS_N",
}


def _claim_status(metrics: dict, required_improvements: list[str], max_worsened_metrics: int = 0) -> tuple[str, list[str], list[str], list[str]]:
    improved = [m for m, data in metrics.items() if data["improved"] is True]
    worsened = [m for m, data in metrics.items() if data["improved"] is False]
    critical = [m for m in ("UnsafeSteps", "ActuatorSaturationRatio") if m in worsened]
    missing_required = [m for m in required_improvements if metrics.get(m, {}).get("improved") is not True]
    supported = not critical and not missing_required and len(worsened) <= max_worsened_metrics
    status = "supported" if supported else "weak_or_contradicted"
    return status, improved, worsened, critical


def _write_config(config: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def _latest_checkpoint(algorithm: str, not_before: float) -> tuple[Path, Path]:
    candidates = sorted(
        (ROOT / "results").glob(f"*_{algorithm}/checkpoints/final.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    candidates = [p for p in candidates if p.stat().st_mtime >= not_before]
    if not candidates:
        raise FileNotFoundError(f"No {algorithm} checkpoint found after {not_before}.")
    checkpoint = candidates[0]
    return checkpoint, checkpoint.parents[1]


def _train_algorithm(config_path: Path, algorithm: str, episodes: int) -> tuple[Path, Path]:
    before = datetime.now().timestamp() - 1.0
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "train_rl.py"),
            "--config",
            str(config_path),
            "--algorithm",
            algorithm,
            "--episodes",
            str(episodes),
        ],
        cwd=ROOT,
        check=True,
    )
    return _latest_checkpoint(algorithm, before)


def _train_variant(config_path: Path, episodes: int) -> tuple[Path, Path]:
    return _train_algorithm(config_path, "ppo", episodes)


def _summarize_metrics(metrics: pd.DataFrame) -> dict:
    out: dict[str, dict[str, float]] = {}
    numeric = metrics.select_dtypes(include="number")
    for controller in metrics["Controller"].unique():
        view = metrics[metrics["Controller"] == controller]
        view_numeric = view[numeric.columns.intersection(view.columns)]
        out[str(controller)] = {}
        for key, value in view_numeric.mean(numeric_only=True).to_dict().items():
            if pd.notna(value):
                out[str(controller)][key] = float(value)
    return out


def _metric_delta(metric: str, candidate: float, baseline: float) -> dict:
    if pd.isna(candidate) or pd.isna(baseline):
        return {"candidate": None, "baseline": None, "delta": None, "relative_delta": None, "improved": None}
    delta = float(candidate - baseline)
    denom = max(abs(float(baseline)), 1e-9)
    relative = float(delta / denom)
    if abs(delta) <= 1e-12:
        improved = None
    else:
        improved = delta < 0.0 if metric in LOWER_IS_BETTER else delta > 0.0
    return {
        "candidate": float(candidate),
        "baseline": float(baseline),
        "delta": delta,
        "relative_delta": relative,
        "improved": improved if improved is None else bool(improved),
    }


def build_claim_report(combined: pd.DataFrame, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    comparisons = [
        {
            "name": "projection_aware_imitation",
            "candidate": "bc_ppo",
            "baseline": "ppo_scratch",
            "claim": "Safe-teacher behavior cloning with projection-aware PPO should outperform standard PPO under the same learned-policy safety projection.",
            "required_metrics": [
                "EpisodeReturn",
                "UnsafeSteps",
                "BodyAccRMS_mps2",
                "PitchAccRMS_radps2",
                "RollAccRMS_radps2",
                "ActionDeltaRMS_N",
                "ActuatorTrackingRMS_N",
                "ActuatorSaturationRatio",
                "PolicyProjectionError",
            ],
            "required_improvements": ["EpisodeReturn", "UnsafeSteps", "ActionDeltaRMS_N"],
            "max_worsened_metrics": 1,
        },
        {
            "name": "imitation_initialization",
            "candidate": "bc_ppo",
            "baseline": "ppo_scratch",
            "claim": "Offline behavior cloning should improve early PPO performance over random initialization.",
            "required_metrics": ["EpisodeReturn", "UnsafeSteps", "ActuatorSaturationRatio", "ActionDeltaRMS_N"],
            "required_improvements": ["EpisodeReturn", "ActionDeltaRMS_N"],
            "max_worsened_metrics": 0,
        },
        {
            "name": "residual_prior_structure",
            "candidate": "residual_bc_ppo",
            "baseline": "bc_ppo",
            "claim": "Residual learning around the full-car prior should improve control quality over direct BC-PPO.",
            "required_metrics": ["EpisodeReturn", "BodyAccRMS_mps2", "PitchAccRMS_radps2", "RollAccRMS_radps2", "UnsafeSteps"],
            "required_improvements": ["EpisodeReturn"],
            "max_worsened_metrics": 1,
        },
        {
            "name": "safe_residual_gate",
            "candidate": "safe_residual_bc_ppo",
            "baseline": "residual_bc_ppo",
            "claim": "Preview/safety-aware residual gating and deviation regularization should reduce unsafe or infeasible residual actions.",
            "required_metrics": ["UnsafeSteps", "ActuatorSaturationRatio", "ActionDeltaRMS_N", "ActionDeviationRMS_N", "EpisodeReturn"],
            "required_improvements": ["ActionDeltaRMS_N", "ActionDeviationRMS_N"],
            "max_worsened_metrics": 0,
        },
    ]
    if combined.empty or "Variant" not in combined.columns or "Controller" not in combined.columns:
        report = {
            "status": "missing_data",
            "reason": "combined metrics are empty or do not contain Variant/Controller columns",
            "comparisons": [{**item, "status": "missing_data", "metrics": {}} for item in comparisons],
        }
    else:
        available = [m for m in CLAIM_METRICS if m in combined.columns]
        grouped = combined.groupby(["Variant", "Controller"], dropna=False)[available].mean(numeric_only=True)

        def row_key(variant: str, controller: str) -> tuple[str, str]:
            return (variant, controller.upper())

        def has_row(variant: str, controller: str) -> bool:
            return row_key(variant, controller) in grouped.index

        def metric_value(variant: str, controller: str, metric: str) -> float:
            return grouped.loc[row_key(variant, controller), metric]

        comparison_results = []
        for item in comparisons:
            candidate = item["candidate"]
            baseline = item["baseline"]
            missing = [name for name in (candidate, baseline) if not has_row(name, "PPO")]
            if missing:
                comparison_results.append({**item, "status": "missing_variant", "missing": missing, "metrics": {}})
                continue
            metrics = {}
            required = [m for m in item["required_metrics"] if m in grouped.columns]
            for metric in required:
                metrics[metric] = _metric_delta(metric, metric_value(candidate, "PPO", metric), metric_value(baseline, "PPO", metric))
            status, improved, worsened, critical = _claim_status(
                metrics,
                list(item.get("required_improvements", [])),
                int(item.get("max_worsened_metrics", 0)),
            )
            comparison_results.append(
                {
                    **item,
                    "status": status,
                    "improved_metrics": improved,
                    "worsened_metrics": worsened,
                    "critical_worsened_metrics": critical,
                    "metrics": metrics,
                }
            )
        for algorithm in ("td3", "sac"):
            baseline_variant = f"{algorithm}_baseline"
            item = {
                "name": f"safe_residual_ppo_vs_{algorithm}",
                "candidate": "safe_residual_bc_ppo",
                "baseline": baseline_variant,
                "claim": f"Safe residual BC-PPO should be competitive with the off-policy {algorithm.upper()} baseline.",
                "required_metrics": [
                    "EpisodeReturn",
                    "UnsafeSteps",
                    "BodyAccRMS_mps2",
                    "PitchAccRMS_radps2",
                    "RollAccRMS_radps2",
                    "ActionDeltaRMS_N",
                    "ActuatorSaturationRatio",
                ],
                "required_improvements": ["EpisodeReturn", "UnsafeSteps", "ActuatorSaturationRatio"],
                "max_worsened_metrics": 2,
            }
            if not has_row("safe_residual_bc_ppo", "PPO") or not has_row(baseline_variant, algorithm):
                missing = []
                if not has_row("safe_residual_bc_ppo", "PPO"):
                    missing.append("safe_residual_bc_ppo/PPO")
                if not has_row(baseline_variant, algorithm):
                    missing.append(f"{baseline_variant}/{algorithm.upper()}")
                comparison_results.append({**item, "status": "missing_variant", "missing": missing, "metrics": {}})
                continue
            metrics = {}
            required = [m for m in item["required_metrics"] if m in grouped.columns]
            for metric in required:
                metrics[metric] = _metric_delta(
                    metric,
                    metric_value("safe_residual_bc_ppo", "PPO", metric),
                    metric_value(baseline_variant, algorithm, metric),
                )
            status, improved, worsened, critical = _claim_status(
                metrics,
                list(item.get("required_improvements", [])),
                int(item.get("max_worsened_metrics", 0)),
            )
            comparison_results.append(
                {
                    **item,
                    "status": status,
                    "improved_metrics": improved,
                    "worsened_metrics": worsened,
                    "critical_worsened_metrics": critical,
                    "metrics": metrics,
                }
            )
        report = {
            "status": "ready" if all(r["status"] != "missing_variant" for r in comparison_results) else "incomplete",
            "core_status": next(
                (r["status"] for r in comparison_results if r.get("name") == "projection_aware_imitation"),
                "missing_data",
            ),
            "controller": "PPO",
            "metric_direction": {
                "higher_is_better": ["EpisodeReturn"],
                "lower_is_better": sorted(LOWER_IS_BETTER),
            },
            "comparisons": comparison_results,
        }

    json_path = out_dir / "si_rppo_claim_report.json"
    md_path = out_dir / "si_rppo_claim_report.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# SI-RPPO Claim Evidence Report",
        "",
        f"- Status: `{report['status']}`",
        f"- Core PPO claim status: `{report.get('core_status', report['status'])}`",
        "- Controller compared: `PPO`",
        "",
        "## Interpretation Rules",
        "",
        "- `EpisodeReturn`: higher is better.",
        "- Safety, comfort, actuator, saturation, and deviation metrics: lower is better.",
        "- A claim is weak if a critical safety/feasibility metric gets worse.",
        "",
        "## Ablation Claims",
        "",
    ]
    for item in report.get("comparisons", []):
        lines += [
            f"### {item['name']}",
            "",
            f"- Claim: {item['claim']}",
            f"- Candidate: `{item['candidate']}`",
            f"- Baseline: `{item['baseline']}`",
            f"- Status: `{item['status']}`",
            "",
        ]
        if item.get("missing"):
            lines.append(f"- Missing variants: `{', '.join(item['missing'])}`")
            lines.append("")
            continue
        if item.get("metrics"):
            lines.append("| Metric | Candidate | Baseline | Delta | RelativeDelta | Improved |")
            lines.append("| --- | --- | --- | --- | --- | --- |")
            for metric, data in item["metrics"].items():
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            metric,
                            "" if data["candidate"] is None else f"{data['candidate']:.6g}",
                            "" if data["baseline"] is None else f"{data['baseline']:.6g}",
                            "" if data["delta"] is None else f"{data['delta']:.6g}",
                            "" if data["relative_delta"] is None else f"{data['relative_delta']:.6g}",
                            str(data["improved"]),
                        ]
                    )
                    + " |"
                )
            lines.append("")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    report["json_path"] = str(json_path.resolve())
    report["markdown_path"] = str(md_path.resolve())
    return report


def run_si_rppo_ablation(
    base_config_path: Path,
    out_dir: Path,
    episodes: int,
    expert_episodes: int,
    expert_max_steps: int | None,
    expert_controller: str = "FULL_CAR_MPC_LITE",
    skip_unsafe_expert: bool = False,
    train_scenario_limit: int | None = None,
    eval_scenario_limit: int | None = None,
    episode_seconds: float | None = None,
    mujoco_settle_seconds: float | None = None,
    variants: list[str] | None = None,
    baseline_algorithms: list[str] | None = None,
    dry_run: bool = False,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    base_config = load_config(base_config_path)
    if train_scenario_limit is not None:
        base_config["scenarios"] = list(base_config["scenarios"])[: int(train_scenario_limit)]
    if episode_seconds is not None:
        base_config["episode_seconds"] = float(episode_seconds)
    if mujoco_settle_seconds is not None:
        base_config.setdefault("mujoco", {})["settle_seconds"] = float(mujoco_settle_seconds)
    base_config["imitation"] = deep_update(
        dict(base_config.get("imitation", {})),
        {
            "dataset": str((out_dir / "expert_dataset.npz").resolve()),
        },
    )
    materialized_base_config = out_dir / "base_config.yaml"
    _write_config(base_config, materialized_base_config)
    dataset_path = out_dir / "expert_dataset.npz"
    if dry_run:
        expert_manifest = {
            "planned": True,
            "out_path": str(dataset_path.resolve()),
            "expert": expert_controller,
            "episodes": int(expert_episodes),
            "max_steps": expert_max_steps,
            "residual_prior": str(base_config.get("residual_control", {}).get("prior", "FULL_CAR_MPC_LITE")),
            "skip_unsafe": bool(skip_unsafe_expert),
        }
    else:
        expert_manifest = collect_expert_dataset(
            config_path=materialized_base_config,
            out_path=dataset_path,
            expert=expert_controller,
            episodes=expert_episodes,
            max_steps=expert_max_steps,
            residual_prior=str(base_config.get("residual_control", {}).get("prior", "FULL_CAR_MPC_LITE")),
            skip_unsafe=skip_unsafe_expert,
        )

    selected = variants or list(VARIANTS)
    variant_reports = {}
    combined_rows = []
    for name in selected:
        if name not in VARIANTS:
            raise ValueError(f"Unknown SI-RPPO ablation variant: {name}")
        variant_dir = out_dir / name
        variant_config = deep_update(deepcopy(base_config), VARIANTS[name])
        variant_config["imitation"] = deep_update(
            dict(variant_config.get("imitation", {})),
            {"dataset": str((out_dir / "expert_dataset.npz").resolve())},
        )
        if episodes <= 1:
            variant_config["imitation"]["epochs"] = min(1, int(variant_config.get("imitation", {}).get("epochs", 1)))
            variant_config["imitation"]["batch_size"] = min(8, int(variant_config.get("imitation", {}).get("batch_size", 8)))
            variant_config["rl"]["eval_every"] = 0
        config_path = variant_dir / "config.yaml"
        _write_config(variant_config, config_path)
        if dry_run:
            variant_reports[name] = {
                "config": str(config_path.resolve()),
                "planned_train_command": [
                    sys.executable,
                    str(ROOT / "scripts" / "train_rl.py"),
                    "--config",
                    str(config_path.resolve()),
                    "--algorithm",
                    "ppo",
                    "--episodes",
                    str(episodes),
                ],
            }
            continue

        checkpoint, run_dir = _train_variant(config_path, episodes)

        eval_config = deepcopy(variant_config)
        if eval_scenario_limit is not None:
            eval_config["scenarios"] = list(eval_config["scenarios"])[: int(eval_scenario_limit)]
        eval_dir = variant_dir / "evaluation"
        metrics, _ = evaluate_all(eval_config, checkpoints={"ppo": str(checkpoint)}, result_dir=eval_dir)
        report_path = build_report(eval_dir)
        metrics_copy = metrics.copy()
        metrics_copy.insert(0, "Variant", name)
        combined_rows.append(metrics_copy)
        variant_reports[name] = {
            "config": str(config_path.resolve()),
            "run_dir": str(run_dir.resolve()),
            "checkpoint": str(checkpoint.resolve()),
            "evaluation_dir": str(eval_dir.resolve()),
            "benchmark_report": str(report_path.resolve()),
            "metric_rows": int(len(metrics)),
            "summary": _summarize_metrics(metrics),
        }

    baseline_reports = {}
    for algorithm in baseline_algorithms or []:
        algorithm = algorithm.lower().strip()
        if algorithm not in {"td3", "sac"}:
            raise ValueError(f"Unsupported off-policy baseline algorithm: {algorithm}")
        name = f"{algorithm}_baseline"
        baseline_dir = out_dir / name
        baseline_config = deep_update(
            deepcopy(base_config),
            {
                "residual_control": {"enabled": False},
                "imitation": {"enabled": False},
                "reward": {"deviation": 0.0},
                "evaluation": {"controllers": ["PASSIVE", "FULL_CAR_MPC_LITE"]},
            },
        )
        if episodes <= 1:
            baseline_config["rl"]["eval_every"] = 0
            baseline_config["rl"]["warmup_steps"] = min(1, int(baseline_config["rl"].get("warmup_steps", 1)))
            baseline_config["rl"]["batch_size"] = min(8, int(baseline_config["rl"].get("batch_size", 8)))
        config_path = baseline_dir / "config.yaml"
        _write_config(baseline_config, config_path)
        if dry_run:
            baseline_reports[name] = {
                "algorithm": algorithm,
                "config": str(config_path.resolve()),
                "planned_train_command": [
                    sys.executable,
                    str(ROOT / "scripts" / "train_rl.py"),
                    "--config",
                    str(config_path.resolve()),
                    "--algorithm",
                    algorithm,
                    "--episodes",
                    str(episodes),
                ],
            }
            continue

        checkpoint, run_dir = _train_algorithm(config_path, algorithm, episodes)
        eval_config = deepcopy(baseline_config)
        if eval_scenario_limit is not None:
            eval_config["scenarios"] = list(eval_config["scenarios"])[: int(eval_scenario_limit)]
        eval_dir = baseline_dir / "evaluation"
        metrics, _ = evaluate_all(eval_config, checkpoints={algorithm: str(checkpoint)}, result_dir=eval_dir)
        report_path = build_report(eval_dir)
        metrics_copy = metrics.copy()
        metrics_copy.insert(0, "Variant", name)
        combined_rows.append(metrics_copy)
        baseline_reports[name] = {
            "algorithm": algorithm,
            "config": str(config_path.resolve()),
            "run_dir": str(run_dir.resolve()),
            "checkpoint": str(checkpoint.resolve()),
            "evaluation_dir": str(eval_dir.resolve()),
            "benchmark_report": str(report_path.resolve()),
            "metric_rows": int(len(metrics)),
            "summary": _summarize_metrics(metrics),
        }

    combined = pd.concat(combined_rows, ignore_index=True) if combined_rows else pd.DataFrame()
    combined_path = out_dir / "combined_metrics.csv"
    combined.to_csv(combined_path, index=False)
    claim_report = build_claim_report(combined, out_dir)
    manifest = {
        "base_config": str(base_config_path.resolve()),
        "materialized_base_config": str(materialized_base_config.resolve()),
        "out_dir": str(out_dir.resolve()),
        "dry_run": bool(dry_run),
        "episodes": int(episodes),
        "episode_seconds": base_config.get("episode_seconds"),
        "mujoco_settle_seconds": base_config.get("mujoco", {}).get("settle_seconds"),
        "expert_controller": expert_controller,
        "skip_unsafe_expert": bool(skip_unsafe_expert),
        "expert_manifest": expert_manifest,
        "variants": variant_reports,
        "off_policy_baselines": baseline_reports,
        "combined_metrics": str(combined_path.resolve()),
        "claim_report": {
            "status": claim_report["status"],
            "core_status": claim_report.get("core_status", claim_report["status"]),
            "json": claim_report["json_path"],
            "markdown": claim_report["markdown_path"],
        },
    }
    (out_dir / "si_rppo_ablation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mujoco_full_car_safe_ppo.yaml")
    parser.add_argument("--out", default=None)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--expert-episodes", type=int, default=4)
    parser.add_argument("--expert-max-steps", type=int, default=None)
    parser.add_argument("--expert-controller", default="FULL_CAR_MPC_LITE")
    parser.add_argument("--skip-unsafe-expert", action="store_true")
    parser.add_argument("--train-scenario-limit", type=int, default=None)
    parser.add_argument("--eval-scenario-limit", type=int, default=None)
    parser.add_argument("--episode-seconds", type=float, default=None)
    parser.add_argument("--mujoco-settle-seconds", type=float, default=None)
    parser.add_argument("--variants", default="", help="Comma-separated subset of ppo_scratch,bc_ppo,residual_bc_ppo,safe_residual_bc_ppo.")
    parser.add_argument("--baseline-algorithms", default="", help="Comma-separated off-policy baselines to include: td3,sac.")
    parser.add_argument("--dry-run", action="store_true", help="Write ablation configs and manifest without collecting data or training.")
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    out_dir = Path(args.out) if args.out else ROOT / "results" / datetime.now().strftime("%Y%m%d_%H%M%S_si_rppo_ablation")
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    variants = [item.strip() for item in args.variants.split(",") if item.strip()] or None
    baseline_algorithms = [item.strip() for item in args.baseline_algorithms.split(",") if item.strip()] or None
    manifest = run_si_rppo_ablation(
        base_config_path=config_path,
        out_dir=out_dir,
        episodes=args.episodes,
        expert_episodes=args.expert_episodes,
        expert_max_steps=args.expert_max_steps,
        expert_controller=args.expert_controller,
        skip_unsafe_expert=args.skip_unsafe_expert,
        train_scenario_limit=args.train_scenario_limit,
        eval_scenario_limit=args.eval_scenario_limit,
        episode_seconds=args.episode_seconds,
        mujoco_settle_seconds=args.mujoco_settle_seconds,
        variants=variants,
        baseline_algorithms=baseline_algorithms,
        dry_run=args.dry_run,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
