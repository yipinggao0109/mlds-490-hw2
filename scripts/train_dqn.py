#!/usr/bin/env python3
"""Train DQN on CartPole-v1 or MsPacman-v5."""

from __future__ import annotations

import argparse
from pathlib import Path

from mlds490_hw2.dqn import DQNConfig, train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=("cartpole", "mspacman"), required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--seed", type=int, default=490)
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--replay-capacity", type=int)
    parser.add_argument("--learning-starts", type=int)
    parser.add_argument("--train-frequency", type=int)
    parser.add_argument("--target-update-frequency", type=int)
    parser.add_argument("--target-tau", type=float)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float)
    parser.add_argument("--epsilon-decay-steps", type=int)
    parser.add_argument("--frame-stack", type=int, default=4)
    parser.add_argument("--reward-clip", type=float)
    parser.add_argument("--terminal-penalty", type=float)
    parser.add_argument("--max-grad-norm", type=float, default=10.0)
    parser.add_argument("--checkpoint-every", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--stop-score", type=float)
    parser.add_argument("--minimum-episodes-before-stop", type=int, default=150)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    defaults = {
        "cartpole": {
            "learning_rate": 5e-4,
            "batch_size": 64,
            "replay_capacity": 50_000,
            "learning_starts": 1_000,
            "train_frequency": 1,
            "target_update_frequency": 500,
            "target_tau": 0.005,
            "terminal_penalty": -10.0,
            "epsilon_end": 0.02,
            "epsilon_decay_steps": 30_000,
            "checkpoint_every": 25,
            "stop_score": 490.0,
        },
        "mspacman": {
            "learning_rate": 1e-4,
            "batch_size": 32,
            "replay_capacity": 20_000,
            "learning_starts": 5_000,
            "train_frequency": 4,
            "target_update_frequency": 5_000,
            "target_tau": None,
            "terminal_penalty": None,
            "epsilon_end": 0.05,
            "epsilon_decay_steps": 500_000,
            "checkpoint_every": 25,
            "stop_score": None,
        },
    }[args.env]

    def selected(name: str):
        value = getattr(args, name)
        return defaults[name] if value is None else value

    config = DQNConfig(
        env=args.env,
        episodes=args.episodes,
        seed=args.seed,
        gamma=args.gamma,
        learning_rate=selected("learning_rate"),
        batch_size=selected("batch_size"),
        replay_capacity=selected("replay_capacity"),
        learning_starts=selected("learning_starts"),
        train_frequency=selected("train_frequency"),
        target_update_frequency=selected("target_update_frequency"),
        target_tau=selected("target_tau"),
        epsilon_start=args.epsilon_start,
        epsilon_end=selected("epsilon_end"),
        epsilon_decay_steps=selected("epsilon_decay_steps"),
        frame_stack=args.frame_stack,
        reward_clip=args.reward_clip,
        terminal_penalty=(
            defaults["terminal_penalty"]
            if args.terminal_penalty is None
            else args.terminal_penalty
        ),
        max_grad_norm=args.max_grad_norm,
        checkpoint_every=selected("checkpoint_every"),
        device=args.device,
        stop_score=selected("stop_score"),
        minimum_episodes_before_stop=args.minimum_episodes_before_stop,
    )
    run_dir = args.output_dir / f"{args.env}_dqn_seed{args.seed}"
    train(config, run_dir, resume=args.resume)


if __name__ == "__main__":
    main()
