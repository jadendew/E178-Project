#!/usr/bin/env python3
"""
create_comparison_animations.py

Creates three side-by-side/in-scene flight comparison animations from CSV logs:

1. MPC v1 vs MPC v2
2. MPC v1 vs Base PID
3. MPC v2 vs Base PID

Each comparison shows both aircraft flying in the same 3D scene.
The "reference/older" aircraft is drawn with lower opacity as a ghost:
    - Base PID is ghosted in comparisons where Base PID appears.
    - MPC v1 is ghosted in the MPC v1 vs MPC v2 comparison.

Outputs:
    comparison_animations/
        mpc_v1_vs_mpc_v2/
            mpc_v1_vs_mpc_v2.mp4
            mpc_v1_vs_mpc_v2.gif
        mpc_v1_vs_base_pid/
            mpc_v1_vs_base_pid.mp4
            mpc_v1_vs_base_pid.gif
        mpc_v2_vs_base_pid/
            mpc_v2_vs_base_pid.mp4
            mpc_v2_vs_base_pid.gif

Required CSV columns:
    t, x, y, z, yaw_deg, pitch_deg, roll_deg

Optional:
    lowerqualitymesh.stl

If the STL is missing or numpy-stl is unavailable, the script uses a simple
built-in aircraft wedge shape.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# =============================================================================
# User settings
# =============================================================================

CSV_FILES = {
    "Base PID": "base_PID_plots/sim_out_basePID.csv",
    "MPC v1": "mpc_output_plots_v1/sim_out_MLP_MPC_v1.csv",
    "MPC v2": "mpc_output_plots_v2/sim_out_MLP_MPC_v2.csv",
}

COMPARISONS = [
    {
        "name": "mpc_v1_vs_mpc_v2",
        "title": "MPC v1 vs MPC v2",
        "primary": "MPC v2",
        "ghost": "MPC v1",
    },
    {
        "name": "mpc_v1_vs_base_pid",
        "title": "MPC v1 vs Base PID",
        "primary": "MPC v1",
        "ghost": "Base PID",
    },
    {
        "name": "mpc_v2_vs_base_pid",
        "title": "MPC v2 vs Base PID",
        "primary": "MPC v2",
        "ghost": "Base PID",
    },
]

OUTPUT_ROOT = "comparison_animations"

# STL settings
STL_PATH = "lowerqualitymesh.stl"
STL_SCALE = 0.001
STL_ROTATION_OFFSET_DEG = (0.0, 0.0, -180.0)
MAX_STL_TRIANGLES = 1200

# Animation settings
STRIDE = 10
FPS = 30
DPI_MP4 = 140
DPI_GIF = 110
TRAIL_LEN = 220
AXIS_LEN = 1.0

# Camera/view settings
FOLLOW_MODE = "midpoint"  # "midpoint" or "fixed"
ZOOM = 42.0               # larger = closer view for follow mode
FIXED_PADDING = 8.0       # only used if FOLLOW_MODE = "fixed"
VIEW_ELEV = 22
VIEW_AZIM = -55

# Visual settings
PRIMARY_ALPHA = 0.90
GHOST_ALPHA = 0.25
TRAIL_ALPHA_PRIMARY = 0.90
TRAIL_ALPHA_GHOST = 0.35
SHOW_BODY_AXES = True
SHOW_LEGEND = True

# Colors; change these if desired.
PRIMARY_COLOR = "tab:blue"
GHOST_COLOR = "0.55"
MPC_V1_COLOR = "tab:orange"
MPC_V2_COLOR = "tab:blue"
BASE_PID_COLOR = "0.45"


# =============================================================================
# Helpers
# =============================================================================

def required_columns() -> list[str]:
    return ["t", "x", "y", "z", "yaw_deg", "pitch_deg", "roll_deg"]


def load_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Could not find CSV: {path}")

    df = pd.read_csv(path)
    missing = [c for c in required_columns() if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    df = df[required_columns()].copy()
    df = df.dropna(subset=required_columns()).reset_index(drop=True)
    df = df.sort_values("t").drop_duplicates(subset=["t"], keep="last")
    df = df.reset_index(drop=True)

    return df


def resample_to_common_time(df_a: pd.DataFrame, df_b: pd.DataFrame, stride: int) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """
    Interpolate both logs onto a common time vector so the aircraft move together.
    """
    t0 = max(float(df_a["t"].min()), float(df_b["t"].min()))
    t1 = min(float(df_a["t"].max()), float(df_b["t"].max()))

    if t1 <= t0:
        raise ValueError("CSV logs do not overlap in time.")

    # Use the median dt of the slower file, then apply stride.
    dt_a = np.median(np.diff(df_a["t"].to_numpy()))
    dt_b = np.median(np.diff(df_b["t"].to_numpy()))
    base_dt = max(float(dt_a), float(dt_b))
    dt_anim = base_dt * max(1, int(stride))

    t_common = np.arange(t0, t1 + 0.5 * dt_anim, dt_anim)

    def interp_df(df: pd.DataFrame) -> pd.DataFrame:
        out = {"t": t_common}
        t_src = df["t"].to_numpy(dtype=float)
        for col in ["x", "y", "z", "yaw_deg", "pitch_deg", "roll_deg"]:
            out[col] = np.interp(t_common, t_src, df[col].to_numpy(dtype=float))
        return pd.DataFrame(out)

    return interp_df(df_a), interp_df(df_b), t_common


def Rx(phi: float) -> np.ndarray:
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[1.0, 0.0, 0.0],
                     [0.0, c, -s],
                     [0.0, s, c]], dtype=float)


def Ry(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0.0, s],
                     [0.0, 1.0, 0.0],
                     [-s, 0.0, c]], dtype=float)


def Rz(psi: float) -> np.ndarray:
    c, s = np.cos(psi), np.sin(psi)
    return np.array([[c, -s, 0.0],
                     [s, c, 0.0],
                     [0.0, 0.0, 1.0]], dtype=float)


def body_to_ned_rotation(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    yaw, pitch, roll = np.deg2rad([yaw_deg, pitch_deg, roll_deg])
    return Rz(yaw) @ Ry(pitch) @ Rx(roll)


def stl_alignment_rotation(offset_deg) -> np.ndarray:
    roll_o, pitch_o, yaw_o = np.deg2rad(offset_deg)
    return Rz(yaw_o) @ Ry(pitch_o) @ Rx(roll_o)


def make_fallback_aircraft_mesh() -> np.ndarray:
    """
    Simple aircraft-like body in body coordinates.
    Shape: nose, wings, tail.
    Returns triangle array with shape (n_triangles, 3, 3).
    """
    nose = np.array([1.00, 0.00, 0.00])
    tail = np.array([-0.70, 0.00, 0.00])
    left_wing = np.array([-0.15, -0.95, 0.05])
    right_wing = np.array([-0.15, 0.95, 0.05])
    top = np.array([-0.20, 0.00, -0.16])
    bottom = np.array([-0.20, 0.00, 0.16])
    tail_l = np.array([-0.75, -0.25, 0.05])
    tail_r = np.array([-0.75, 0.25, 0.05])

    tris = np.array([
        [nose, right_wing, top],
        [nose, top, left_wing],
        [nose, bottom, right_wing],
        [nose, left_wing, bottom],
        [tail, top, right_wing],
        [tail, left_wing, top],
        [tail, right_wing, bottom],
        [tail, bottom, left_wing],
        [tail, tail_r, top],
        [tail, top, tail_l],
    ], dtype=float)
    return tris


def load_body_mesh() -> np.ndarray:
    """
    Load STL triangles and convert into body-frame meters.
    Falls back to a simple internal mesh if STL loading fails.
    """
    stl_path = Path(STL_PATH)

    if stl_path.is_file():
        try:
            from stl import mesh

            stl_mesh = mesh.Mesh.from_file(str(stl_path))
            vectors = stl_mesh.vectors.copy().astype(float)

            if vectors.shape[0] > MAX_STL_TRIANGLES:
                keep_idx = np.linspace(0, vectors.shape[0] - 1, MAX_STL_TRIANGLES).astype(int)
                vectors = vectors[keep_idx]
                print(f"Downsampled STL from {stl_mesh.vectors.shape[0]} to {vectors.shape[0]} triangles.")

            # Center, scale, and align.
            vectors -= vectors.reshape(-1, 3).mean(axis=0)
            vectors *= float(STL_SCALE)

            R_offset = stl_alignment_rotation(STL_ROTATION_OFFSET_DEG)
            vectors = np.einsum("ij,tkj->tki", R_offset, vectors)

            return vectors

        except Exception as e:
            print(f"Could not load STL '{stl_path}': {e}")
            print("Using fallback aircraft mesh.")

    else:
        print(f"STL not found at '{stl_path}'. Using fallback aircraft mesh.")

    return make_fallback_aircraft_mesh()


def ned_to_plot(points_ned: np.ndarray) -> np.ndarray:
    """
    NED to plotting coordinates with Z-up:
        x_plot = x_ned
        y_plot = y_ned
        z_plot = -z_ned
    """
    T = np.diag([1.0, 1.0, -1.0])
    return np.einsum("ij,...j->...i", T, points_ned)


def transform_mesh_to_plot(mesh_body: np.ndarray, row: pd.Series) -> np.ndarray:
    pos_ned = np.array([row["x"], row["y"], row["z"]], dtype=float)
    R_bw = body_to_ned_rotation(row["yaw_deg"], row["pitch_deg"], row["roll_deg"])

    mesh_ned = np.einsum("ij,tkj->tki", R_bw, mesh_body)
    mesh_ned = mesh_ned + pos_ned
    return ned_to_plot(mesh_ned)


def set_axes_equal_around(ax, center: np.ndarray, radius: float):
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def get_color(label: str, is_ghost: bool) -> str:
    if is_ghost:
        return GHOST_COLOR
    if label == "MPC v1":
        return MPC_V1_COLOR
    if label == "MPC v2":
        return MPC_V2_COLOR
    if label == "Base PID":
        return BASE_PID_COLOR
    return PRIMARY_COLOR


def make_comparison_animation(
    df_primary: pd.DataFrame,
    df_ghost: pd.DataFrame,
    primary_label: str,
    ghost_label: str,
    title: str,
    out_dir: Path,
    out_name: str,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    df_primary, df_ghost, t_common = resample_to_common_time(df_primary, df_ghost, STRIDE)

    mesh_body = load_body_mesh()

    primary_color = get_color(primary_label, is_ghost=False)
    ghost_color = get_color(ghost_label, is_ghost=True)

    x1, y1, z1 = df_primary["x"].to_numpy(), df_primary["y"].to_numpy(), -df_primary["z"].to_numpy()
    x2, y2, z2 = df_ghost["x"].to_numpy(), df_ghost["y"].to_numpy(), -df_ghost["z"].to_numpy()

    all_xyz = np.column_stack([np.r_[x1, x2], np.r_[y1, y2], np.r_[z1, z2]])
    fixed_center = np.mean(all_xyz, axis=0)
    fixed_radius = 0.5 * np.max(np.ptp(all_xyz, axis=0)) + FIXED_PADDING
    fixed_radius = max(float(fixed_radius), 5.0)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(title)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z up (m)")
    ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)

    primary_mesh = Poly3DCollection(transform_mesh_to_plot(mesh_body, df_primary.iloc[0]), alpha=PRIMARY_ALPHA)
    primary_mesh.set_facecolor(primary_color)
    primary_mesh.set_edgecolor("k")
    primary_mesh.set_linewidth(0.2)

    ghost_mesh = Poly3DCollection(transform_mesh_to_plot(mesh_body, df_ghost.iloc[0]), alpha=GHOST_ALPHA)
    ghost_mesh.set_facecolor(ghost_color)
    ghost_mesh.set_edgecolor("k")
    ghost_mesh.set_linewidth(0.15)

    ax.add_collection3d(ghost_mesh)
    ax.add_collection3d(primary_mesh)

    primary_trail, = ax.plot([], [], [], color=primary_color, linewidth=2.0, alpha=TRAIL_ALPHA_PRIMARY, label=primary_label)
    ghost_trail, = ax.plot([], [], [], color=ghost_color, linewidth=1.6, linestyle="--", alpha=TRAIL_ALPHA_GHOST, label=f"{ghost_label} ghost")

    primary_dot, = ax.plot([], [], [], marker="o", color=primary_color, markersize=4, linestyle="")
    ghost_dot, = ax.plot([], [], [], marker="o", color=ghost_color, markersize=4, linestyle="", alpha=GHOST_ALPHA)

    axis_lines = []
    if SHOW_BODY_AXES:
        for color, alpha in [(primary_color, 0.9), (ghost_color, 0.35)]:
            for _ in range(3):
                line, = ax.plot([], [], [], color=color, alpha=alpha, linewidth=1.4)
                axis_lines.append(line)

    time_text = ax.text2D(0.02, 0.96, "", transform=ax.transAxes)

    if SHOW_LEGEND:
        ax.legend(loc="upper left")

    def update_body_axes(row, start_idx: int, alpha_scale: float):
        if not SHOW_BODY_AXES:
            return

        pos_ned = np.array([row["x"], row["y"], row["z"]], dtype=float)
        pos_plot = ned_to_plot(pos_ned)
        R_bw = body_to_ned_rotation(row["yaw_deg"], row["pitch_deg"], row["roll_deg"])

        body_axes = [
            np.array([1.0, 0.0, 0.0]),   # body +X
            np.array([0.0, 1.0, 0.0]),   # body +Y
            np.array([0.0, 0.0, -1.0]),  # body +Z converted visually upward convention
        ]

        for j, axis_b in enumerate(body_axes):
            tip_plot = pos_plot + AXIS_LEN * ned_to_plot(R_bw @ axis_b)
            line = axis_lines[start_idx + j]
            line.set_data([pos_plot[0], tip_plot[0]], [pos_plot[1], tip_plot[1]])
            line.set_3d_properties([pos_plot[2], tip_plot[2]])

    def update(i: int):
        row_p = df_primary.iloc[i]
        row_g = df_ghost.iloc[i]

        primary_mesh.set_verts(transform_mesh_to_plot(mesh_body, row_p))
        ghost_mesh.set_verts(transform_mesh_to_plot(mesh_body, row_g))

        i0 = max(0, i - int(TRAIL_LEN))

        primary_trail.set_data(x1[i0:i + 1], y1[i0:i + 1])
        primary_trail.set_3d_properties(z1[i0:i + 1])

        ghost_trail.set_data(x2[i0:i + 1], y2[i0:i + 1])
        ghost_trail.set_3d_properties(z2[i0:i + 1])

        primary_dot.set_data([x1[i]], [y1[i]])
        primary_dot.set_3d_properties([z1[i]])

        ghost_dot.set_data([x2[i]], [y2[i]])
        ghost_dot.set_3d_properties([z2[i]])

        if SHOW_BODY_AXES:
            update_body_axes(row_p, 0, 1.0)
            update_body_axes(row_g, 3, GHOST_ALPHA)

        if FOLLOW_MODE.lower() == "midpoint":
            center = np.array([
                0.5 * (x1[i] + x2[i]),
                0.5 * (y1[i] + y2[i]),
                0.5 * (z1[i] + z2[i]),
            ], dtype=float)

            # Use local separation and global motion to set a stable view radius.
            local_sep = np.linalg.norm(np.array([x1[i] - x2[i], y1[i] - y2[i], z1[i] - z2[i]]))
            radius = max(4.0, 0.5 * local_sep + 10.0)
            radius = radius / max(float(ZOOM) / 42.0, 1e-6)
            set_axes_equal_around(ax, center, radius)
        else:
            set_axes_equal_around(ax, fixed_center, fixed_radius)

        time_text.set_text(f"t = {t_common[i]:.2f} s")

        artists = [
            primary_mesh, ghost_mesh,
            primary_trail, ghost_trail,
            primary_dot, ghost_dot,
            time_text,
        ] + axis_lines
        return artists

    anim = FuncAnimation(
        fig,
        update,
        frames=len(t_common),
        interval=1000.0 / FPS,
        blit=False,
        cache_frame_data=False,
    )

    mp4_path = out_dir / f"{out_name}.mp4"
    gif_path = out_dir / f"{out_name}.gif"

    print(f"Saving MP4: {mp4_path}")
    try:
        anim.save(mp4_path, writer=FFMpegWriter(fps=FPS, bitrate=1800), dpi=DPI_MP4)
        print(f"Saved MP4: {mp4_path}")
    except Exception as e:
        print(f"MP4 save failed for {out_name}: {e}")

    print(f"Saving GIF: {gif_path}")
    try:
        anim.save(gif_path, writer=PillowWriter(fps=FPS), dpi=DPI_GIF)
        print(f"Saved GIF: {gif_path}")
    except Exception as e:
        print(f"GIF save failed for {out_name}: {e}")

    plt.close(fig)


def main():
    root = Path(__file__).resolve().parent
    os.chdir(root)

    dfs = {}
    for label, csv_path in CSV_FILES.items():
        dfs[label] = load_csv(csv_path)
        print(f"Loaded {label}: {csv_path}, rows={len(dfs[label])}")

    output_root = Path(OUTPUT_ROOT)
    output_root.mkdir(parents=True, exist_ok=True)

    for comp in COMPARISONS:
        primary = comp["primary"]
        ghost = comp["ghost"]
        out_name = comp["name"]
        out_dir = output_root / out_name

        print("\n" + "=" * 80)
        print(f"Creating comparison: {comp['title']}")
        print(f"Primary: {primary}")
        print(f"Ghost:   {ghost}")
        print("=" * 80)

        make_comparison_animation(
            df_primary=dfs[primary],
            df_ghost=dfs[ghost],
            primary_label=primary,
            ghost_label=ghost,
            title=comp["title"],
            out_dir=out_dir,
            out_name=out_name,
        )

    print("\nDone. Animations saved under:")
    print(output_root.resolve())


if __name__ == "__main__":
    main()
