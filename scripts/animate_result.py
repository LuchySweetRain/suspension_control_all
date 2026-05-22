from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from visualization.animation import animate_half_car


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Trajectory CSV from an evaluation result directory")
    parser.add_argument("--out", default=None, help="Output .gif or video path")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--vertical-scale", type=float, default=20.0)
    args = parser.parse_args()

    out = animate_half_car(
        args.csv,
        out_path=args.out,
        fps=args.fps,
        stride=args.stride,
        vertical_scale=args.vertical_scale,
    )
    print(f"Saved animation to {out}")


if __name__ == "__main__":
    main()
