#!/usr/bin/env python3
"""Record the first decisions of a trained DQN for validation."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from mlds490_hw2.dqn import (
    FrameStack,
    load_run,
    make_environment,
    next_state_from_observation,
    reset_state,
    select_action,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1490)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint", default="best_checkpoint.pt")
    args = parser.parse_args()

    config, network, device = load_run(args.run_dir, args.device, args.checkpoint)
    environment = make_environment(config)
    stack = FrameStack(config.frame_stack) if config.env == "mspacman" else None
    generator = np.random.default_rng(args.seed)
    state = reset_state(environment, config, args.seed, stack)
    rows: list[dict[str, float | int]] = []
    cumulative_reward = 0.0
    try:
        for step in range(args.steps):
            action, max_q, q_values = select_action(
                network, state, 0.0, config.n_actions, generator, device
            )
            observation, reward, terminated, truncated, _ = environment.step(action)
            cumulative_reward += float(reward)
            row: dict[str, float | int] = {
                "step": step,
                "action": action,
                "max_q": max_q,
                "reward": float(reward),
                "cumulative_reward": cumulative_reward,
                "terminated": int(terminated),
                "truncated": int(truncated),
            }
            for index, value in enumerate(q_values):
                row[f"q_action_{index}"] = float(value)
            if config.env == "cartpole":
                row.update(
                    {
                        "cart_position": float(state[0]),
                        "cart_velocity": float(state[1]),
                        "pole_angle": float(state[2]),
                        "pole_angular_velocity": float(state[3]),
                    }
                )
            else:
                row["state_min"] = int(np.min(state))
                row["state_max"] = int(np.max(state))
                row["nonzero_pixels"] = int(np.count_nonzero(state))
            rows.append(row)
            state = next_state_from_observation(observation, config, stack)
            if terminated or truncated:
                break
    finally:
        environment.close()

    frame = pd.DataFrame(rows)
    output_path = args.run_dir / f"first_{args.steps}_steps_trace.csv"
    frame.to_csv(output_path, index=False)
    print(frame.to_string(index=False))
    print(f"saved={output_path}")


if __name__ == "__main__":
    main()
