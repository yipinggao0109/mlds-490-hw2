#!/usr/bin/env python3
"""Compare a stable target network with an online-copy ablation on CartPole."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mlds490_hw2.dqn import moving_average


def summarize(path: Path, episode_limit: int = 300) -> dict[str, float | int | str]:
    history = pd.read_csv(path / "training_history.csv").iloc[:episode_limit]
    rewards = history["reward"].to_numpy(dtype=np.float64)
    smoothed = moving_average(rewards, 100)
    return {
        "run": str(path),
        "episodes": len(history),
        "first_100_mean_reward": float(np.mean(rewards[:100])),
        "last_100_mean_reward": float(np.mean(rewards[-100:])),
        "best_100_mean_reward": float(np.nanmax(smoothed)),
        "last_100_mean_max_q": float(history["mean_max_q"].tail(100).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stable-run", type=Path, required=True)
    parser.add_argument("--no-lag-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/validation"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    stable = pd.read_csv(args.stable_run / "training_history.csv").iloc[:300]
    no_lag = pd.read_csv(args.no_lag_run / "training_history.csv").iloc[:300]
    summaries = [summarize(args.stable_run), summarize(args.no_lag_run)]
    (args.output_dir / "target_network_ablation.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )
    pd.DataFrame(summaries).to_csv(args.output_dir / "target_network_ablation.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for history, label, color in (
        (stable, "Soft target update", "#1B6CA8"),
        (no_lag, "No lagged target", "#C44536"),
    ):
        rewards = history["reward"].to_numpy(dtype=np.float64)
        axes[0].plot(
            history["episode"], moving_average(rewards, 100), label=label, color=color
        )
        axes[1].plot(history["episode"], history["mean_max_q"], label=label, color=color)
    axes[0].set(
        xlabel="Training episode",
        ylabel="100-episode moving-average reward",
        title="Reward stability",
    )
    axes[1].set(
        xlabel="Training episode",
        ylabel="Mean maximum Q-value",
        title="Q-value scale",
        yscale="symlog",
    )
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend()
    fig.suptitle("CartPole target-network ablation")
    fig.tight_layout()
    fig.savefig(args.output_dir / "target_network_ablation.png", dpi=200)
    fig.savefig(args.output_dir / "target_network_ablation.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
