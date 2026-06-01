from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_METRICS = [
    "EpisodeReturn",
    "UnsafeSteps",
    "BodyAccRMS_mps2",
    "PitchAccRMS_radps2",
    "RollAccRMS_radps2",
    "ActionDeltaRMS_N",
    "ActuatorTrackingRMS_N",
    "PolicyProjectionError",
]


def _read_seed_summary(path: Path) -> pd.DataFrame:
    data = json.loads((path / "projection_seed_sweep_summary.json").read_text(encoding="utf-8"))
    rows = []
    for row in data.get("seeds", []):
        out = {
            "Source": path.name,
            "Seed": int(row["seed"]),
            "CoreStatus": row.get("core_status"),
        }
        for metric in DEFAULT_METRICS:
            out[f"ppo_scratch_{metric}"] = row.get(f"ppo_scratch_{metric}")
            out[f"bc_ppo_{metric}"] = row.get(f"bc_ppo_{metric}")
            out[f"delta_{metric}"] = row.get(f"delta_{metric}")
        rows.append(out)
    return pd.DataFrame(rows)


def _read_full_matrix(path: Path) -> pd.DataFrame:
    metrics = pd.read_csv(path / "combined_metrics.csv")
    available = [m for m in DEFAULT_METRICS + ["ActuatorSaturationRatio", "ActionDeviationRMS_N"] if m in metrics.columns]
    grouped = metrics.groupby(["Variant", "Controller"], dropna=False)[available].mean(numeric_only=True).reset_index()
    grouped.insert(0, "Source", path.name)
    return grouped


def _write_markdown(seed_df: pd.DataFrame, matrix_df: pd.DataFrame, out_path: Path) -> None:
    lines = [
        "# Delta-Parameterized PPO Evidence Table",
        "",
        "## Repeated Seed Core Claim",
        "",
        "| Seed | CoreStatus | ReturnDelta | UnsafeDelta | BodyDelta | PitchDelta | RollDelta | ActionDeltaDelta | TrackingDelta | ProjectionErrorDelta |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in seed_df.sort_values("Seed").iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(int(row["Seed"])),
                    str(row["CoreStatus"]),
                    f"{row['delta_EpisodeReturn']:.4f}",
                    f"{row['delta_UnsafeSteps']:.4f}",
                    f"{row['delta_BodyAccRMS_mps2']:.4f}",
                    f"{row['delta_PitchAccRMS_radps2']:.4f}",
                    f"{row['delta_RollAccRMS_radps2']:.4f}",
                    f"{row['delta_ActionDeltaRMS_N']:.4f}",
                    f"{row['delta_ActuatorTrackingRMS_N']:.4f}",
                    f"{row['delta_PolicyProjectionError']:.4f}",
                ]
            )
            + " |"
        )
    supported = int((seed_df["CoreStatus"] == "supported").sum()) if not seed_df.empty else 0
    lines += [
        "",
        f"Supported seeds: {supported} / {len(seed_df)}.",
        "",
        "## Full Matrix Mean Metrics",
        "",
        "| Variant | Controller | Return | Unsafe | Body | Pitch | Roll | ActionDelta | Tracking | ProjectionError |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in matrix_df.sort_values(["Variant", "Controller"]).iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["Variant"]),
                    str(row["Controller"]),
                    f"{row.get('EpisodeReturn', float('nan')):.4f}",
                    f"{row.get('UnsafeSteps', float('nan')):.4f}",
                    f"{row.get('BodyAccRMS_mps2', float('nan')):.4f}",
                    f"{row.get('PitchAccRMS_radps2', float('nan')):.4f}",
                    f"{row.get('RollAccRMS_radps2', float('nan')):.4f}",
                    f"{row.get('ActionDeltaRMS_N', float('nan')):.4f}",
                    f"{row.get('ActuatorTrackingRMS_N', float('nan')):.4f}",
                    "" if pd.isna(row.get("PolicyProjectionError")) else f"{row.get('PolicyProjectionError'):.4f}",
                ]
            )
            + " |"
        )
    lines += [
        "",
        "## Reading",
        "",
        "- `ReturnDelta` is BC-PPO minus PPO-scratch; higher is better.",
        "- Other deltas are BC-PPO minus PPO-scratch; lower is better.",
        "- The full matrix includes passive and full-car MPC-lite reference rows because each variant evaluation reports all configured controllers.",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def build_delta_ppo_evidence_table(seed_dirs: list[Path], full_matrix_dir: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_df = pd.concat([_read_seed_summary(path) for path in seed_dirs], ignore_index=True)
    matrix_df = _read_full_matrix(full_matrix_dir)
    seed_csv = out_dir / "delta_ppo_seed_evidence.csv"
    matrix_csv = out_dir / "delta_ppo_full_matrix.csv"
    md_path = out_dir / "delta_ppo_evidence_table.md"
    seed_df.to_csv(seed_csv, index=False)
    matrix_df.to_csv(matrix_csv, index=False)
    _write_markdown(seed_df, matrix_df, md_path)
    summary = {
        "seed_dirs": [str(path.resolve()) for path in seed_dirs],
        "full_matrix_dir": str(full_matrix_dir.resolve()),
        "supported_seeds": int((seed_df["CoreStatus"] == "supported").sum()) if not seed_df.empty else 0,
        "seed_count": int(len(seed_df)),
        "seed_csv": str(seed_csv.resolve()),
        "matrix_csv": str(matrix_csv.resolve()),
        "markdown": str(md_path.resolve()),
    }
    (out_dir / "delta_ppo_evidence_table.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-dirs", nargs="+", required=True)
    parser.add_argument("--full-matrix-dir", required=True)
    parser.add_argument("--out", default="results/delta_ppo_evidence_table")
    args = parser.parse_args()
    summary = build_delta_ppo_evidence_table(
        seed_dirs=[Path(item) for item in args.seed_dirs],
        full_matrix_dir=Path(args.full_matrix_dir),
        out_dir=Path(args.out),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
