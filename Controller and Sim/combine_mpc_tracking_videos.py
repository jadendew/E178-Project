#!/usr/bin/env python3
"""
combine_mpc_tracking_videos.py

Combines two MP4s into one synchronized side-by-side MP4, then exports a GIF
from that combined MP4.

Inputs:
    mpc_v2_animation.mp4
    attitude_tracking_live_MPC_v2.mp4

Outputs:
    combined_mpc_tracking/
        mpc_v2_flight_and_tracking_combined.mp4
        mpc_v2_flight_and_tracking_combined.gif

Why this works:
    Once the two animations are rendered into one single video/GIF file, the
    flight view and tracking plot cannot drift out of sync in the browser.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np


# =============================================================================
# User settings
# =============================================================================

LEFT_VIDEO = "mpc_output_plots_v2/mpc_output_animation_v2/mpc_v2_animation.mp4"
RIGHT_VIDEO = "mpc_output_plots_v2/attitude_tracking_animation/attitude_tracking_live_MPC_v2.mp4"

OUTPUT_DIR = "mpc_output_plots_v2/combined_mpc_tracking"
OUTPUT_MP4 = "mpc_v2_flight_and_tracking_combined.mp4"
OUTPUT_GIF = "mpc_v2_flight_and_tracking_combined.gif"

# Combined layout
FPS_OUT = 30

# The two panels will be resized to this height before being placed side-by-side.
# 720 is website-friendly. Use 900 or 1080 for higher quality/larger files.
PANEL_HEIGHT = 720

# Width fraction for the left flight panel.
# 0.50 = equal width left/right.
LEFT_WIDTH_FRAC = 0.50

# Fill behavior:
#   "stretch" = force each video to fill its panel exactly, no whitespace, may distort aspect ratio
#   "contain" = preserve aspect ratio, add padding if needed
#   "cover"   = preserve aspect ratio, crop if needed
FIT_MODE = "stretch"

# Visual styling
GAP_PX = 18
BORDER_PX = 0
BACKGROUND_RGB = (245, 245, 245)

# GIF settings
GIF_FPS = 15
GIF_SCALE_WIDTH = 1280
GIF_DITHER = "bayer"  # "bayer" usually cleaner for plots than default dithering


# =============================================================================
# Helpers
# =============================================================================

def open_video(path: str | Path):
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Could not find video: {path}")

    cap = cv2.VideoCapture(str(path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    info = {
        "path": path,
        "fps": float(cap.get(cv2.CAP_PROP_FPS)),
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }

    info["duration"] = info["frames"] / max(info["fps"], 1e-9)

    return cap, info


def resize_stretch(frame: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    return cv2.resize(frame, (int(target_w), int(target_h)), interpolation=cv2.INTER_AREA)


def resize_contain(frame: np.ndarray, target_w: int, target_h: int, bg_rgb=BACKGROUND_RGB) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = min(target_w / w, target_h / h)

    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.full((target_h, target_w, 3), bg_rgb[::-1], dtype=np.uint8)
    x0 = (target_w - new_w) // 2
    y0 = (target_h - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized

    return canvas


def resize_cover(frame: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = max(target_w / w, target_h / h)

    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    x0 = max(0, (new_w - target_w) // 2)
    y0 = max(0, (new_h - target_h) // 2)

    return resized[y0:y0 + target_h, x0:x0 + target_w]


def fit_frame(frame: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    mode = FIT_MODE.lower().strip()

    if mode == "stretch":
        return resize_stretch(frame, target_w, target_h)

    if mode == "contain":
        return resize_contain(frame, target_w, target_h)

    if mode == "cover":
        return resize_cover(frame, target_w, target_h)

    raise ValueError(f"Unknown FIT_MODE: {FIT_MODE}")


def write_combined_mp4():
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    left_cap, left_info = open_video(LEFT_VIDEO)
    right_cap, right_info = open_video(RIGHT_VIDEO)

    print("Left video:")
    print(f"  {left_info['path']}")
    print(f"  {left_info['width']}x{left_info['height']}, {left_info['fps']:.3f} fps, {left_info['duration']:.3f} s")

    print("Right video:")
    print(f"  {right_info['path']}")
    print(f"  {right_info['width']}x{right_info['height']}, {right_info['fps']:.3f} fps, {right_info['duration']:.3f} s")

    duration = min(left_info["duration"], right_info["duration"])
    n_frames = int(np.floor(duration * FPS_OUT))

    out_h = int(PANEL_HEIGHT)
    usable_w = int(round(out_h * 16 / 9 * 2))  # default wide combined base width
    left_w = int(round(usable_w * LEFT_WIDTH_FRAC))
    right_w = int(usable_w - left_w)

    out_w = left_w + right_w + int(GAP_PX) + 2 * int(BORDER_PX)

    mp4_path = out_dir / OUTPUT_MP4

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(mp4_path), fourcc, float(FPS_OUT), (out_w, out_h))

    if not writer.isOpened():
        raise RuntimeError("Could not open output MP4 writer. Try installing/updating OpenCV or changing codec.")

    bg_bgr = BACKGROUND_RGB[::-1]

    for i in range(n_frames):
        t = i / float(FPS_OUT)

        left_cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        right_cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)

        ok_l, frame_l = left_cap.read()
        ok_r, frame_r = right_cap.read()

        if not ok_l or not ok_r:
            print(f"Stopped early at frame {i}/{n_frames}")
            break

        left_panel = fit_frame(frame_l, left_w, out_h)
        right_panel = fit_frame(frame_r, right_w, out_h)

        gap = np.full((out_h, int(GAP_PX), 3), bg_bgr, dtype=np.uint8)

        combined = np.concatenate([left_panel, gap, right_panel], axis=1)

        if BORDER_PX > 0:
            combined = cv2.copyMakeBorder(
                combined,
                top=0,
                bottom=0,
                left=int(BORDER_PX),
                right=int(BORDER_PX),
                borderType=cv2.BORDER_CONSTANT,
                value=bg_bgr,
            )

        writer.write(combined)

        if i % max(1, n_frames // 20) == 0:
            print(f"  writing frame {i:5d}/{n_frames}")

    writer.release()
    left_cap.release()
    right_cap.release()

    print(f"Saved combined MP4: {mp4_path.resolve()}")

    return mp4_path


def make_gif_from_mp4(mp4_path: Path):
    gif_path = Path(OUTPUT_DIR) / OUTPUT_GIF

    # Prefer ffmpeg if available because it gives much better GIF quality.
    palette_path = gif_path.with_suffix(".palette.png")

    vf_palette = (
        f"fps={GIF_FPS},"
        f"scale={GIF_SCALE_WIDTH}:-1:flags=lanczos,"
        f"palettegen"
    )

    vf_gif = (
        f"fps={GIF_FPS},"
        f"scale={GIF_SCALE_WIDTH}:-1:flags=lanczos[x];"
        f"[x][1:v]paletteuse=dither={GIF_DITHER}"
    )

    try:
        print("Generating GIF palette with ffmpeg...")
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i", str(mp4_path),
                "-vf", vf_palette,
                str(palette_path),
            ],
            check=True,
        )

        print("Generating GIF with ffmpeg...")
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i", str(mp4_path),
                "-i", str(palette_path),
                "-lavfi", vf_gif,
                str(gif_path),
            ],
            check=True,
        )

        if palette_path.exists():
            palette_path.unlink()

        print(f"Saved combined GIF: {gif_path.resolve()}")
        return gif_path

    except Exception as e:
        print(f"ffmpeg GIF export failed: {e}")
        print("Falling back to OpenCV + imageio. Quality/file size may be worse.")

    try:
        import imageio.v2 as imageio

        cap = cv2.VideoCapture(str(mp4_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open combined MP4 for GIF export: {mp4_path}")

        source_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = max(1, int(round(source_fps / GIF_FPS)))

        frames = []
        idx = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if idx % frame_interval == 0:
                h, w = frame.shape[:2]
                new_w = int(GIF_SCALE_WIDTH)
                new_h = int(round(h * new_w / w))

                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)

            idx += 1

        cap.release()

        imageio.mimsave(gif_path, frames, fps=GIF_FPS)
        print(f"Saved combined GIF: {gif_path.resolve()}")
        return gif_path

    except Exception as e:
        print(f"OpenCV/imageio GIF export failed: {e}")
        print("MP4 was still created successfully.")
        return None


def main():
    mp4_path = write_combined_mp4()
    make_gif_from_mp4(mp4_path)


if __name__ == "__main__":
    main()
