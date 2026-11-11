#!/usr/bin/env python3
"""Evaluate a trained DQN and plot its reward distribution."""

from __future__ import annotations

import argparse
from pathlib import Path

from mlds490_hw2.dqn import evaluate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--checkpoint", default="best_checkpoint.pt")
    args = parser.parse_args()
    evaluate(args.run_dir, args.episodes, args.epsilon, args.device, args.checkpoint)


if __name__ == "__main__":
    main()
