# MLDS 490 Assignment 2

This repository contains a reproducible Deep Q-Network implementation for CartPole-v1 and
MsPacman-v5. The implementation uses a replay buffer, a target network, Double DQN targets,
Huber loss, gradient clipping, and a linear epsilon schedule.

## Environment setup

Python 3.11 or 3.12 is recommended. From the repository root, run:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e '.[dev]'
AutoROM --accept-license
```

## Quick validation

```bash
pytest
ruff check .
```

## Selected training runs

```bash
python scripts/train_dqn.py --env cartpole --episodes 800 --device cpu \
  --learning-rate 0.0005 --batch-size 128 --replay-capacity 10000 \
  --learning-starts 1000 --epsilon-decay-steps 10000 --target-tau 0.005 \
  --terminal-penalty -10 --output-dir results/terminal_penalty

python scripts/train_dqn.py --env mspacman --episodes 4000 --device mps \
  --learning-rate 0.0001 --batch-size 32 --replay-capacity 20000 \
  --learning-starts 5000 --train-frequency 4 \
  --target-update-frequency 5000 --epsilon-decay-steps 500000 \
  --checkpoint-every 25 --reward-clip 1 --output-dir results
```

Each run records its configuration, episode history, action counts, checkpoints, and figures in a
separate directory under `results/`. Training can be continued with `--resume`.

## Evaluation

```bash
python scripts/evaluate_dqn.py \
  --run-dir results/terminal_penalty/cartpole_dqn_seed490 \
  --episodes 500 --device cpu
python scripts/evaluate_dqn.py \
  --run-dir results/mspacman_dqn_seed490 --episodes 500 --device mps
```

## Quantitative validation

```bash
python scripts/evaluate_random_policy.py --env cartpole --episodes 500
python scripts/evaluate_random_policy.py --env mspacman --episodes 500
python scripts/trace_dqn.py \
  --run-dir results/terminal_penalty/cartpole_dqn_seed490 --steps 10
python scripts/trace_dqn.py --run-dir results/mspacman_dqn_seed490 --steps 10

# Target-network ablation used in the report
python scripts/train_dqn.py --env cartpole --episodes 300 --device cpu \
  --learning-rate 0.0005 --batch-size 128 --replay-capacity 10000 \
  --learning-starts 1000 --epsilon-decay-steps 10000 --target-tau 1.0 \
  --terminal-penalty -10 --output-dir results/ablation_no_lag
python scripts/compare_target_ablation.py \
  --stable-run results/terminal_penalty/cartpole_dqn_seed490 \
  --no-lag-run results/ablation_no_lag/cartpole_dqn_seed490
```

The action-frequency heatmap is the non-standard plot. It shows how the action distribution changes
from exploration to the learned policy over blocks of training episodes.

The CartPole learner replaces the reward stored for a true early termination with -10. The CSV and
plots always retain the original environment episode reward. This penalty makes failure visible within
the shorter effective horizon produced by the required discount factor of 0.95.
