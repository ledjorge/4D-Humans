#!/usr/bin/env python
"""
Visualize tracked SMPL trajectories on a 2D plane.

The PHALP tracker stores per-frame results as a pickled dictionary
(`outputs/results/<name>.pkl`). Each entry contains, among other things,
the predicted SMPL joints for the tracked identities. This script loads
that dictionary, extracts the XY coordinates of a chosen SMPL joint (the
pelvis by default), and plots the trajectory of each track in a shared
2D plot.

Example:
    python scripts/plot_track_trajectories.py \\
        outputs/results/demo_IMG_1517_frames.pkl \\
        --output trajectory.png
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np


def _collect_tracks(
    frames: Dict[str, Dict],
    joint_index: int,
    min_length: int,
) -> Dict[int, Dict[str, List[float]]]:
    """Bundle XY positions for each tracked identity."""
    tracks: Dict[int, Dict[str, List[float]]] = {}

    # Sort frames by their timestamp so the trajectories are ordered.
    ordered_frames = sorted(frames.items(), key=lambda item: item[1]["time"])
    for frame_name, payload in ordered_frames:
        tracked_ids = payload.get("tracked_ids", [])
        joints = payload.get("3d_joints", [])

        if not tracked_ids or not joints:
            continue

        for idx, track_id in enumerate(tracked_ids):
            if idx >= len(joints):
                # Should not happen, but guard against malformed data.
                continue

            joint_array = np.asarray(joints[idx])
            if joint_array.ndim != 2 or joint_index >= joint_array.shape[0]:
                continue

            x, y = joint_array[joint_index, :2]
            track = tracks.setdefault(
                track_id, {"x": [], "y": [], "frames": [], "names": []}
            )
            track["x"].append(float(x))
            track["y"].append(float(y))
            track["frames"].append(int(payload.get("time", 0)))
            track["names"].append(frame_name)

    # Optionally drop very short tracks.
    if min_length > 1:
        tracks = {
            tid: data
            for tid, data in tracks.items()
            if len(data["x"]) >= min_length
        }

    return tracks


def _plot_tracks(
    tracks: Dict[int, Dict[str, List[float]]],
    title: str,
    output: Path | None,
) -> None:
    """Plot the XY trajectories."""
    if not tracks:
        raise RuntimeError("No tracks available after filtering.")

    plt.figure(figsize=(8, 8))
    color_cycle = itertools.cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])

    for track_id, data in tracks.items():
        color = next(color_cycle)
        plt.plot(data["x"], data["y"], marker="o", linestyle="-", color=color, label=f"ID {track_id}")

        # Annotate start & end to help debugging.
        plt.text(data["x"][0], data["y"][0], f"{track_id}s", color=color, fontsize=8)
        plt.text(data["x"][-1], data["y"][-1], f"{track_id}e", color=color, fontsize=8)

    plt.xlabel("X position (SMPL joint units)")
    plt.ylabel("Y position (SMPL joint units)")
    plt.title(title)
    plt.legend(loc="best", fontsize=8)
    plt.axis("equal")
    plt.grid(True, linestyle="--", alpha=0.3)

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output, bbox_inches="tight", dpi=200)
        print(f"Saved trajectory plot to {output}")
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "pkl_path",
        type=Path,
        help="Path to the tracking results pickle (outputs/results/*.pkl).",
    )
    parser.add_argument(
        "--joint-index",
        type=int,
        default=0,
        help="SMPL joint index to use for XY coordinates (default: 0 / pelvis).",
    )
    parser.add_argument(
        "--min-track-length",
        type=int,
        default=1,
        help="Minimum number of frames a track must have to be plotted.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output PNG path. If omitted, the plot window pops up.",
    )
    args = parser.parse_args()

    if not args.pkl_path.exists():
        raise FileNotFoundError(f"Tracking file not found: {args.pkl_path}")

    frames = joblib.load(args.pkl_path)
    if not isinstance(frames, dict):
        raise TypeError(f"Unexpected pickle payload type: {type(frames)!r}")

    tracks = _collect_tracks(frames, args.joint_index, args.min_track_length)
    title = f"SMPL Trajectories ({args.pkl_path.stem})"
    _plot_tracks(tracks, title, args.output)


if __name__ == "__main__":
    main()
