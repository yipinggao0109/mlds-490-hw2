#!/usr/bin/env python3
"""Plot the learned CartPole action preference over pole states."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from mlds490_hw2.dqn import load_run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint", default="best_checkpoint.pt")
    args = parser.parse_args()

    config, network, device = load_run(args.run_dir, args.device, args.checkpoint)
    if config.env != "cartpole":
        raise ValueError("this diagnostic is defined only for CartPole")

    angles = np.linspace(-0.21, 0.21, 101)
    angular_velocities = np.linspace(-2.5, 2.5, 101)
    angle_grid, velocity_grid = np.meshgrid(angles, angular_velocities)
    states = np.column_stack(
        [
            np.zeros(angle_grid.size),
            np.zeros(angle_grid.size),
            angle_grid.ravel(),
            velocity_grid.ravel(),
        ]
    ).astype(np.float32)
    with torch.no_grad():
        q_values = network(torch.as_tensor(states, device=device)).cpu().numpy()
    preference = (q_values[:, 1] - q_values[:, 0]).reshape(angle_grid.shape)

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    limit = float(np.max(np.abs(preference)))
    image = ax.imshow(
        preference,
        extent=[angles.min(), angles.max(), angular_velocities.min(), angular_velocities.max()],
        origin="lower",
        aspect="auto",
        cmap="coolwarm",
        vmin=-limit,
        vmax=limit,
    )
    ax.contour(angle_grid, velocity_grid, preference, levels=[0], colors="black", linewidths=1.5)
    ax.set(
        xlabel="Pole angle (radians)",
        ylabel="Pole angular velocity (radians per second)",
        title="Learned CartPole action preference",
    )
    fig.colorbar(image, ax=ax, label="Q(right) - Q(left)")
    fig.tight_layout()
    fig.savefig(args.run_dir / "cartpole_q_preference.png", dpi=200)
    fig.savefig(args.run_dir / "cartpole_q_preference.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
