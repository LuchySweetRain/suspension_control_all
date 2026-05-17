from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from visualization.plots import plot_all


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True)
    args = parser.parse_args()
    plot_all(Path(args.result_dir))


if __name__ == "__main__":
    main()

