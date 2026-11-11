#!/usr/bin/env python3
"""Evaluate a uniform random policy as a quantitative reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mlds490_hw2.dqn import DQNConfig, make_environment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=("cartpole", "mspacman"), required=True)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=490)
    parser.add_argument("--output-dir", type=Path, default=Path("results/random_policy"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = DQNConfig(env=args.env, episodes=1, seed=args.seed)
    environment = make_environment(config)
    generator = np.random.default_rng(args.seed)
    rows: list[dict[str, float | int]] = []
    try:
        for episode in range(1, args.episodes + 1):
            environment.reset(seed=args.seed + 100_000 + episode)
            terminated = truncated = False
            reward_sum = 0.0
            length = 0
            while not (terminated or truncated):
                action = int(generator.integers(config.n_actions))
                _, reward, terminated, truncated, _ = environment.step(action)
                reward_sum += float(reward)
                length += 1
            rows.append({"episode": episode, "reward": reward_sum, "length": length})
    finally:
        environment.close()

    frame = pd.DataFrame(rows)
    rewards = frame["reward"].to_numpy(dtype=np.float64)
    summary = {
        "environment": config.gym_id,
        "episodes": args.episodes,
        "mean_reward": float(np.mean(rewards)),
        "standard_deviation": float(np.std(rewards, ddof=0)),
        "minimum_reward": float(np.min(rewards)),
        "maximum_reward": float(np.max(rewards)),
        "mean_episode_length": float(frame["length"].mean()),
    }
    prefix = args.output_dir / f"{args.env}_uniform_random"
    frame.to_csv(prefix.with_suffix(".csv"), index=False)
    prefix.with_name(prefix.name + "_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.hist(rewards, bins="auto", color="#7A8B99", edgecolor="white")
    ax.axvline(
        np.mean(rewards), color="#C1121F", linestyle="--", linewidth=2,
        label=f"Mean = {np.mean(rewards):.2f}",
    )
    ax.set(xlabel="Episode reward", ylabel="Count", title=f"Random policy on {config.gym_id}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(prefix.with_suffix(".png"), dpi=200)
    fig.savefig(prefix.with_suffix(".pdf"))
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
