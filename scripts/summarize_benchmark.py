from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PRIMARY_METRICS = [
    "BodyAccRMS_mps2",
    "PitchAccRMS_radps2",
    "RollAccRMS_radps2",
    "MaxFrontSusp_m",
    "MaxRearSusp_m",
    "ActionDeltaRMS_N",
    "CommandDeltaRMS_N",
    "ActuatorTrackingRMS_N",
    "ActuatorSaturationRatio",
    "ActionDeviationRMS_N",
    "UnsafeSteps",
    "EpisodeReturn",
]


def _format_float(value) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (int, float)):
        return f"{float(value):.6g}"
    return str(value)


def build_report(result_dir: Path) -> Path:
    metrics_path = result_dir / "metrics.csv"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Missing metrics.csv in {result_dir}")
    metrics = pd.read_csv(metrics_path)
    manifest_path = result_dir / "evaluation_manifest.json"
    manifest = {}
    if manifest_path.is_file():
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)

    available_metrics = [m for m in PRIMARY_METRICS if m in metrics.columns]
    grouped = metrics.groupby("Controller", dropna=False)[available_metrics].mean(numeric_only=True)
    report_path = result_dir / "benchmark_report.md"
    lines = [
        "# MuJoCo Suspension Benchmark Report",
        "",
        f"- Result directory: `{result_dir}`",
        f"- Engine: `{manifest.get('engine', 'unknown')}`",
        f"- Controllers: `{', '.join(manifest.get('controllers', sorted(metrics['Controller'].unique())))}`",
        f"- Scenarios: `{', '.join(manifest.get('scenarios', sorted(metrics['Scenario'].unique())))}`",
        f"- Trajectories: `{len(manifest.get('trajectory_files', [])) or len(list(result_dir.glob('*.csv'))) - 1}`",
        "",
        "## Mean Metrics By Controller",
        "",
    ]
    header = ["Controller", *available_metrics]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for controller, row in grouped.iterrows():
        lines.append("| " + " | ".join([str(controller), *[_format_float(row[m]) for m in available_metrics]]) + " |")

    lines += [
        "",
        "## Per-Scenario Metrics",
        "",
    ]
    per_cols = ["Controller", "Scenario", *available_metrics]
    lines.append("| " + " | ".join(per_cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(per_cols)) + " |")
    for _, row in metrics.sort_values(["Scenario", "Controller"]).iterrows():
        lines.append(
            "| "
            + " | ".join([str(row["Controller"]), str(row["Scenario"]), *[_format_float(row[m]) for m in available_metrics]])
            + " |"
        )

    figure_dir = result_dir / "figures"
    figures = sorted(figure_dir.glob("*.png")) if figure_dir.is_dir() else []
    if figures:
        lines += ["", "## Figures", ""]
        for figure in figures:
            rel = figure.relative_to(result_dir)
            lines.append(f"- [{rel.as_posix()}]({rel.as_posix()})")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True)
    args = parser.parse_args()
    report_path = build_report(Path(args.result_dir))
    print(f"Saved benchmark report to {report_path}")


if __name__ == "__main__":
    main()
