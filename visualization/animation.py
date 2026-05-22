from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import pandas as pd


def animate_half_car(
    csv_path: str | Path,
    out_path: str | Path | None = None,
    fps: int = 30,
    stride: int = 2,
    vertical_scale: float = 20.0,
) -> Path:
    csv_path = Path(csv_path)
    out_path = Path(out_path) if out_path else csv_path.with_suffix(".gif")
    df = pd.read_csv(csv_path).iloc[:: max(1, int(stride))].reset_index(drop=True)
    if df.empty:
        raise ValueError(f"No samples found in {csv_path}")

    wheelbase = 2.7
    front_x = 1.2
    rear_x = -1.5
    body_half_height = 0.18
    tire_radius = 0.28

    for col in ("zdf", "zdr"):
        if col not in df:
            df[col] = df["zwf" if col == "zdf" else "zwr"] - tire_radius

    z_cols = ["zb", "zwf", "zwr", "zdf", "zdr"]
    z_min = float(df[z_cols].min().min()) * vertical_scale - 0.8
    z_max = float(df[z_cols].max().max()) * vertical_scale + 0.8

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_xlim(rear_x - 0.8, front_x + 0.8)
    ax.set_ylim(z_min, z_max)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("Longitudinal position (m)")
    ax.set_ylabel(f"Vertical displacement x{vertical_scale:g}")

    body_line, = ax.plot([], [], color="black", linewidth=5, solid_capstyle="round")
    road_line, = ax.plot([], [], color="tab:gray", linewidth=3)
    front_link, = ax.plot([], [], color="tab:blue", linewidth=2)
    rear_link, = ax.plot([], [], color="tab:blue", linewidth=2)
    front_wheel = plt.Circle((0, 0), tire_radius * 0.65, fill=False, linewidth=3, color="tab:orange")
    rear_wheel = plt.Circle((0, 0), tire_radius * 0.65, fill=False, linewidth=3, color="tab:orange")
    ax.add_patch(front_wheel)
    ax.add_patch(rear_wheel)
    title = ax.set_title("")

    def body_points(row):
        zb = row.zb * vertical_scale
        theta = row.theta
        x = np.array([rear_x, front_x])
        y = zb + x * np.sin(theta) * vertical_scale
        return x, y

    def update(i: int):
        row = df.iloc[i]
        bx, by = body_points(row)
        wf = row.zwf * vertical_scale
        wr = row.zwr * vertical_scale
        rdf = row.zdf * vertical_scale
        rdr = row.zdr * vertical_scale

        body_line.set_data(bx, by + body_half_height)
        road_line.set_data([rear_x, front_x], [rdr - tire_radius, rdf - tire_radius])
        front_link.set_data([front_x, front_x], [by[1], wf])
        rear_link.set_data([rear_x, rear_x], [by[0], wr])
        front_wheel.center = (front_x, wf)
        rear_wheel.center = (rear_x, wr)
        title.set_text(
            f"{csv_path.stem}  t={row.time:.2f}s  "
            f"Uaf={row.Uaf:.1f}N  Uar={row.Uar:.1f}N"
        )
        return body_line, road_line, front_link, rear_link, front_wheel, rear_wheel, title

    anim = FuncAnimation(fig, update, frames=len(df), interval=1000 / fps, blit=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".gif":
        anim.save(out_path, writer=PillowWriter(fps=fps))
    else:
        anim.save(out_path, fps=fps)
    plt.close(fig)
    return out_path
