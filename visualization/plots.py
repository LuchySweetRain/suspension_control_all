from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


STYLE = {
    "PID": ("tab:blue", "-."),
    "SPDF": ("tab:red", "-"),
    "MPC": ("tab:green", "--"),
    "RL": ("tab:purple", "-"),
    "TD3": ("tab:purple", "-"),
    "DDPG": ("tab:orange", "--"),
    "SAC": ("tab:brown", "-."),
    "PPO": ("tab:pink", ":"),
}


def load_trajectories(result_dir: Path) -> dict[str, pd.DataFrame]:
    trajectories = {}
    for csv_path in result_dir.glob("*.csv"):
        if csv_path.name == "metrics.csv":
            continue
        trajectories[csv_path.stem] = pd.read_csv(csv_path)
    return trajectories


def plot_all(result_dir: str | Path):
    result_dir = Path(result_dir)
    fig_dir = result_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    trajectories = load_trajectories(result_dir)
    if not trajectories:
        raise RuntimeError(f"No trajectory CSV files found in {result_dir}")
    _plot_response(trajectories, fig_dir)
    _plot_control(trajectories, fig_dir)
    _plot_psd(trajectories, fig_dir)


def _plot_response(trajs: dict[str, pd.DataFrame], fig_dir: Path):
    for scenario in sorted({name.split("_", 1)[1] for name in trajs}):
        fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
        for key, df in trajs.items():
            ctrl, sc = key.split("_", 1)
            if sc != scenario:
                continue
            color, ls = STYLE.get(ctrl, ("k", "-"))
            axes[0, 0].plot(df["time"], df["ddzb"], label=ctrl, color=color, linestyle=ls)
            axes[0, 1].plot(df["time"], df["ddtheta"], label=ctrl, color=color, linestyle=ls)
            axes[1, 0].plot(df["time"], df["delta_yf"], label=ctrl, color=color, linestyle=ls)
            axes[1, 1].plot(df["time"], df["delta_yr"], label=ctrl, color=color, linestyle=ls)
        axes[0, 0].set_ylabel("Body acc. (m/s^2)")
        axes[0, 1].set_ylabel("Pitch acc. (rad/s^2)")
        axes[1, 0].set_ylabel("Front susp. (m)")
        axes[1, 1].set_ylabel("Rear susp. (m)")
        for ax in axes.ravel():
            ax.grid(True, alpha=0.3)
            ax.legend()
        fig.suptitle(f"{scenario} response")
        fig.tight_layout()
        fig.savefig(fig_dir / f"{scenario}_response.png", dpi=200)
        plt.close(fig)


def _plot_control(trajs: dict[str, pd.DataFrame], fig_dir: Path):
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for key, df in trajs.items():
        if not key.endswith("class_b"):
            continue
        ctrl = key.split("_", 1)[0]
        color, ls = STYLE.get(ctrl, ("k", "-"))
        axes[0].plot(df["time"], df["Uaf"], label=ctrl, color=color, linestyle=ls)
        axes[1].plot(df["time"], df["Uar"], label=ctrl, color=color, linestyle=ls)
    axes[0].set_ylabel("Uaf (N)")
    axes[1].set_ylabel("Uar (N)")
    axes[1].set_xlabel("Time (s)")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "control_input_class_b.png", dpi=200)
    plt.close(fig)


def _plot_psd(trajs: dict[str, pd.DataFrame], fig_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for key, df in trajs.items():
        if not key.endswith("class_b"):
            continue
        ctrl = key.split("_", 1)[0]
        color, ls = STYLE.get(ctrl, ("k", "-"))
        freq, pxx = _simple_psd(df["time"].to_numpy(), df["ddzb"].to_numpy())
        axes[0].loglog(freq[1:], pxx[1:], label=ctrl, color=color, linestyle=ls)
        freq, pxx = _simple_psd(df["time"].to_numpy(), df["ddtheta"].to_numpy())
        axes[1].loglog(freq[1:], pxx[1:], label=ctrl, color=color, linestyle=ls)
    axes[0].set_title("Body acceleration PSD")
    axes[1].set_title("Pitch acceleration PSD")
    for ax in axes:
        ax.set_xlabel("Frequency (Hz)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "psd_summary_class_b.png", dpi=200)
    plt.close(fig)


def _simple_psd(time: np.ndarray, signal: np.ndarray):
    dt = float(np.mean(np.diff(time)))
    signal = signal - np.mean(signal)
    spec = np.fft.rfft(signal)
    freq = np.fft.rfftfreq(signal.size, dt)
    pxx = (np.abs(spec) ** 2) * dt / max(signal.size, 1)
    return freq, pxx
