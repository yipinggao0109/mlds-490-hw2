"""Deep Q-Network training and evaluation utilities.

The implementation supports CartPole-v1 and ALE/MsPacman-v5. It uses a replay
buffer, a separate target network, Double DQN targets, Huber loss, gradient
clipping, and a linear epsilon schedule.
"""

from __future__ import annotations

import json
import random
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

EnvironmentName = Literal["cartpole", "mspacman"]


@dataclass(frozen=True)
class DQNConfig:
    """Configuration for one DQN training run."""

    env: EnvironmentName
    episodes: int
    seed: int = 490
    gamma: float | None = None
    learning_rate: float = 1e-3
    batch_size: int = 64
    replay_capacity: int = 50_000
    learning_starts: int = 1_000
    train_frequency: int = 1
    target_update_frequency: int = 500
    target_tau: float | None = None
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 50_000
    frame_stack: int = 4
    reward_clip: float | None = None
    terminal_penalty: float | None = None
    max_grad_norm: float = 10.0
    checkpoint_every: int = 25
    device: str = "auto"
    stop_score: float | None = None
    minimum_episodes_before_stop: int = 150

    @property
    def resolved_gamma(self) -> float:
        if self.gamma is not None:
            return self.gamma
        return 0.95 if self.env == "cartpole" else 0.99

    @property
    def gym_id(self) -> str:
        return "CartPole-v1" if self.env == "cartpole" else "ALE/MsPacman-v5"

    @property
    def n_actions(self) -> int:
        return 2 if self.env == "cartpole" else 9

    @property
    def state_shape(self) -> tuple[int, ...]:
        return (4,) if self.env == "cartpole" else (self.frame_stack, 88, 80)

    @property
    def state_dtype(self) -> np.dtype[Any]:
        return np.dtype(np.float32 if self.env == "cartpole" else np.int8)

    def validate(self) -> None:
        if self.episodes <= 0:
            raise ValueError("episodes must be positive")
        if not 0 <= self.resolved_gamma < 1:
            raise ValueError("gamma must lie in [0, 1)")
        if self.learning_rate <= 0 or self.batch_size <= 0:
            raise ValueError("learning rate and batch size must be positive")
        if self.replay_capacity < self.batch_size:
            raise ValueError("replay capacity must be at least the batch size")
        if self.learning_starts < self.batch_size:
            raise ValueError("learning_starts must be at least the batch size")
        if self.train_frequency <= 0 or self.target_update_frequency <= 0:
            raise ValueError("training and target update frequencies must be positive")
        if self.target_tau is not None and not 0 < self.target_tau <= 1:
            raise ValueError("target_tau must lie in (0, 1]")
        if not 0 <= self.epsilon_end <= self.epsilon_start <= 1:
            raise ValueError("epsilon values must satisfy 0 <= end <= start <= 1")
        if self.epsilon_decay_steps <= 0 or self.frame_stack <= 0:
            raise ValueError("epsilon decay steps and frame stack must be positive")


class ReplayBuffer:
    """Fixed-size replay buffer backed by preallocated NumPy arrays."""

    def __init__(
        self,
        capacity: int,
        state_shape: tuple[int, ...],
        state_dtype: np.dtype[Any],
        seed: int,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.states = np.empty((capacity, *state_shape), dtype=state_dtype)
        self.next_states = np.empty((capacity, *state_shape), dtype=state_dtype)
        self.actions = np.empty(capacity, dtype=np.int64)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.terminated = np.empty(capacity, dtype=np.bool_)
        self.position = 0
        self.size = 0
        self.generator = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.size

    def add(
        self,
        state: NDArray[Any],
        action: int,
        reward: float,
        next_state: NDArray[Any],
        terminated: bool,
    ) -> None:
        self.states[self.position] = state
        self.next_states[self.position] = next_state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.terminated[self.position] = terminated
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self, batch_size: int
    ) -> tuple[NDArray[Any], NDArray[np.int64], NDArray[np.float32], NDArray[Any], NDArray[np.bool_]]:
        if batch_size > self.size:
            raise ValueError("cannot sample more transitions than the buffer contains")
        indices = self.generator.choice(self.size, size=batch_size, replace=False)
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.terminated[indices],
        )


def preprocess_mspacman(observation: NDArray[np.uint8]) -> NDArray[np.int8]:
    """Apply the 88 by 80 grayscale preprocessing given in the assignment."""

    if observation.shape != (210, 160, 3):
        raise ValueError(f"expected a 210x160x3 frame, received {observation.shape}")
    image = observation[1:176:2, ::2].sum(axis=2)
    image[image == 210 + 164 + 74] = 0
    return (image // 3 - 128).astype(np.int8)


class FrameStack:
    """Small frame stack that never mixes frames across episode boundaries."""

    def __init__(self, count: int) -> None:
        if count <= 0:
            raise ValueError("frame count must be positive")
        self.frames: deque[NDArray[np.int8]] = deque(maxlen=count)
        self.count = count

    def reset(self, observation: NDArray[np.uint8]) -> NDArray[np.int8]:
        frame = preprocess_mspacman(observation)
        self.frames.clear()
        for _ in range(self.count):
            self.frames.append(frame.copy())
        return self.state()

    def append(self, observation: NDArray[np.uint8]) -> NDArray[np.int8]:
        if not self.frames:
            raise RuntimeError("frame stack must be reset before appending")
        self.frames.append(preprocess_mspacman(observation))
        return self.state()

    def state(self) -> NDArray[np.int8]:
        if len(self.frames) != self.count:
            raise RuntimeError("frame stack is incomplete")
        return np.stack(tuple(self.frames), axis=0)


class CartPoleQNetwork(nn.Module):
    def __init__(self, n_actions: int = 2) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, states: Tensor) -> Tensor:
        return self.network(states.float())


class AtariQNetwork(nn.Module):
    def __init__(self, frame_stack: int = 4, n_actions: int = 9) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(frame_stack, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        with torch.no_grad():
            feature_count = int(self.features(torch.zeros(1, frame_stack, 88, 80)).numel())
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_count, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, states: Tensor) -> Tensor:
        scaled = states.float() / 128.0
        return self.head(self.features(scaled))


def build_network(config: DQNConfig) -> nn.Module:
    if config.env == "cartpole":
        return CartPoleQNetwork(config.n_actions)
    return AtariQNetwork(config.frame_stack, config.n_actions)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def make_environment(config: DQNConfig, render_mode: str | None = None) -> gym.Env[Any, Any]:
    if config.env == "mspacman":
        import ale_py

        gym.register_envs(ale_py)
    environment = gym.make(config.gym_id, render_mode=render_mode)
    if int(environment.action_space.n) != config.n_actions:
        raise RuntimeError(
            f"{config.gym_id} exposed {environment.action_space.n} actions, expected {config.n_actions}"
        )
    environment.action_space.seed(config.seed)
    return environment


def reset_state(
    environment: gym.Env[Any, Any],
    config: DQNConfig,
    episode_seed: int,
    frame_stack: FrameStack | None,
) -> NDArray[Any]:
    observation, _ = environment.reset(seed=episode_seed)
    if config.env == "cartpole":
        return np.asarray(observation, dtype=np.float32)
    if frame_stack is None:
        raise RuntimeError("MsPacman requires a frame stack")
    return frame_stack.reset(np.asarray(observation, dtype=np.uint8))


def next_state_from_observation(
    observation: NDArray[Any], config: DQNConfig, frame_stack: FrameStack | None
) -> NDArray[Any]:
    if config.env == "cartpole":
        return np.asarray(observation, dtype=np.float32)
    if frame_stack is None:
        raise RuntimeError("MsPacman requires a frame stack")
    return frame_stack.append(np.asarray(observation, dtype=np.uint8))


def epsilon_at_step(config: DQNConfig, step: int) -> float:
    fraction = min(max(step, 0) / config.epsilon_decay_steps, 1.0)
    return config.epsilon_start + fraction * (config.epsilon_end - config.epsilon_start)


def select_action(
    network: nn.Module,
    state: NDArray[Any],
    epsilon: float,
    n_actions: int,
    generator: np.random.Generator,
    device: torch.device,
) -> tuple[int, float, NDArray[np.float32]]:
    state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        q_values = network(state_tensor).squeeze(0)
    q_numpy = q_values.detach().cpu().numpy().astype(np.float32)
    if generator.random() < epsilon:
        action = int(generator.integers(n_actions))
    else:
        action = int(np.argmax(q_numpy))
    return action, float(np.max(q_numpy)), q_numpy


def compute_double_dqn_targets(
    online_network: nn.Module,
    target_network: nn.Module,
    next_states: Tensor,
    rewards: Tensor,
    terminated: Tensor,
    gamma: float,
) -> Tensor:
    """Build Double DQN targets, bootstrapping across time-limit truncation."""

    with torch.no_grad():
        next_actions = online_network(next_states).argmax(dim=1, keepdim=True)
        next_q = target_network(next_states).gather(1, next_actions).squeeze(1)
        return rewards + gamma * (1.0 - terminated.float()) * next_q


def optimize_batch(
    online_network: nn.Module,
    target_network: nn.Module,
    optimizer: torch.optim.Optimizer,
    replay_buffer: ReplayBuffer,
    config: DQNConfig,
    device: torch.device,
) -> dict[str, float]:
    states, actions, rewards, next_states, terminated = replay_buffer.sample(config.batch_size)
    state_tensor = torch.as_tensor(states, dtype=torch.float32, device=device)
    action_tensor = torch.as_tensor(actions, dtype=torch.int64, device=device)
    reward_tensor = torch.as_tensor(rewards, dtype=torch.float32, device=device)
    next_state_tensor = torch.as_tensor(next_states, dtype=torch.float32, device=device)
    terminated_tensor = torch.as_tensor(terminated, dtype=torch.float32, device=device)

    predicted = online_network(state_tensor).gather(1, action_tensor[:, None]).squeeze(1)
    targets = compute_double_dqn_targets(
        online_network,
        target_network,
        next_state_tensor,
        reward_tensor,
        terminated_tensor,
        config.resolved_gamma,
    )
    loss = torch.nn.functional.smooth_l1_loss(predicted, targets)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        online_network.parameters(), config.max_grad_norm
    )
    optimizer.step()

    with torch.no_grad():
        td_error = torch.mean(torch.abs(targets - predicted))
        target_gap = torch.mean(
            torch.abs(online_network(state_tensor) - target_network(state_tensor))
        )
    return {
        "loss": float(loss.detach().cpu()),
        "mean_abs_td_error": float(td_error.detach().cpu()),
        "online_target_gap": float(target_gap.detach().cpu()),
        "gradient_norm_before_clipping": float(gradient_norm.detach().cpu()),
    }


def soft_update_target(
    online_network: nn.Module, target_network: nn.Module, tau: float
) -> None:
    """Move target parameters a fraction tau toward the online parameters."""

    if not 0 < tau <= 1:
        raise ValueError("tau must lie in (0, 1]")
    with torch.no_grad():
        for target_parameter, online_parameter in zip(
            target_network.parameters(), online_network.parameters(), strict=True
        ):
            target_parameter.mul_(1.0 - tau).add_(online_parameter, alpha=tau)


def moving_average(values: NDArray[np.float64], window: int = 100) -> NDArray[np.float64]:
    if window <= 0:
        raise ValueError("window must be positive")
    result = np.full(values.shape, np.nan, dtype=np.float64)
    if len(values) >= window:
        result[window - 1 :] = np.convolve(values, np.ones(window) / window, mode="valid")
    return result


def plot_training_history(history: pd.DataFrame, run_dir: Path) -> None:
    rewards = history["reward"].to_numpy(dtype=np.float64)
    smoothed = moving_average(rewards, 100)

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.plot(history["episode"], rewards, color="#8DA9C4", alpha=0.45, linewidth=0.8,
            label="Episode reward")
    ax.plot(history["episode"], smoothed, color="#0B3954", linewidth=2.0,
            label="100-episode moving average")
    ax.set(xlabel="Training episode", ylabel="Episode reward", title="DQN training reward")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "training_reward.png", dpi=200)
    fig.savefig(run_dir / "training_reward.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.plot(history["episode"], history["mean_max_q"], color="#C44536", linewidth=1.3)
    ax.set(
        xlabel="Training episode",
        ylabel="Mean maximum Q-value",
        title="Maximum Q-values during training",
    )
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(run_dir / "maximum_q_values.png", dpi=200)
    fig.savefig(run_dir / "maximum_q_values.pdf")
    plt.close(fig)


def plot_action_frequency(action_counts: NDArray[np.int64], run_dir: Path) -> None:
    """Create a non-standard heatmap showing how action use changes during training."""

    n_episodes, n_actions = action_counts.shape
    n_blocks = min(20, n_episodes)
    block_edges = np.linspace(0, n_episodes, n_blocks + 1, dtype=int)
    frequencies = np.zeros((n_actions, n_blocks), dtype=np.float64)
    for block in range(n_blocks):
        counts = action_counts[block_edges[block] : block_edges[block + 1]].sum(axis=0)
        frequencies[:, block] = counts / max(counts.sum(), 1)

    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    image = ax.imshow(frequencies, aspect="auto", origin="lower", cmap="viridis")
    labels = [
        f"{block_edges[index] + 1}-{block_edges[index + 1]}"
        for index in range(n_blocks)
    ]
    tick_positions = np.arange(n_blocks)[:: max(1, n_blocks // 5)]
    ax.set_xticks(tick_positions, [labels[index] for index in tick_positions], rotation=30)
    ax.set_yticks(np.arange(n_actions))
    ax.set(
        xlabel="Training episode block",
        ylabel="Action",
        title="Action-selection frequency over training",
    )
    fig.colorbar(image, ax=ax, label="Fraction of selected actions")
    fig.tight_layout()
    fig.savefig(run_dir / "action_frequency_heatmap.png", dpi=200)
    fig.savefig(run_dir / "action_frequency_heatmap.pdf")
    plt.close(fig)


def save_checkpoint(
    path: Path,
    online_network: nn.Module,
    target_network: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: DQNConfig,
    completed_episodes: int,
    global_step: int,
    best_moving_average: float,
) -> None:
    torch.save(
        {
            "online_state_dict": online_network.state_dict(),
            "target_state_dict": target_network.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": asdict(config),
            "completed_episodes": completed_episodes,
            "global_step": global_step,
            "best_moving_average": best_moving_average,
        },
        path,
    )


def train(config: DQNConfig, run_dir: Path, resume: bool = False) -> pd.DataFrame:
    """Train one DQN run and save all required training plots and diagnostics."""

    config.validate()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps({**asdict(config), "resolved_gamma": config.resolved_gamma}, indent=2),
        encoding="utf-8",
    )
    seed_everything(config.seed)
    generator = np.random.default_rng(config.seed)
    device = resolve_device(config.device)
    print(f"device={device}", flush=True)

    online_network = build_network(config).to(device)
    target_network = build_network(config).to(device)
    target_network.load_state_dict(online_network.state_dict())
    target_network.eval()
    optimizer = torch.optim.Adam(online_network.parameters(), lr=config.learning_rate)
    replay_buffer = ReplayBuffer(
        config.replay_capacity, config.state_shape, config.state_dtype, config.seed
    )
    environment = make_environment(config)
    stack = FrameStack(config.frame_stack) if config.env == "mspacman" else None

    records: list[dict[str, float | int]] = []
    action_rows: list[NDArray[np.int64]] = []
    global_step = 0
    start_episode = 1
    best_moving_average = -float("inf")
    history_path = run_dir / "training_history.csv"
    actions_path = run_dir / "action_counts.csv"
    checkpoint_path = run_dir / "checkpoint.pt"

    if resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        online_network.load_state_dict(checkpoint["online_state_dict"])
        target_network.load_state_dict(checkpoint["target_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_episode = int(checkpoint["completed_episodes"]) + 1
        global_step = int(checkpoint["global_step"])
        best_moving_average = float(checkpoint.get("best_moving_average", -float("inf")))
        if history_path.exists():
            records = pd.read_csv(history_path).to_dict(orient="records")
        if actions_path.exists():
            action_rows = [row for row in pd.read_csv(actions_path).to_numpy(dtype=np.int64)]

    try:
        for episode in range(start_episode, config.episodes + 1):
            state = reset_state(environment, config, config.seed + episode, stack)
            terminated = truncated = False
            episode_reward = 0.0
            episode_length = 0
            max_q_values: list[float] = []
            metrics: list[dict[str, float]] = []
            action_counts = np.zeros(config.n_actions, dtype=np.int64)

            while not (terminated or truncated):
                epsilon = epsilon_at_step(config, global_step)
                online_network.eval()
                action, max_q, _ = select_action(
                    online_network,
                    state,
                    epsilon,
                    config.n_actions,
                    generator,
                    device,
                )
                observation, reward, terminated, truncated, _ = environment.step(action)
                next_state = next_state_from_observation(observation, config, stack)
                stored_reward = float(reward)
                if terminated and config.terminal_penalty is not None:
                    stored_reward = config.terminal_penalty
                if config.reward_clip is not None:
                    stored_reward = float(np.clip(stored_reward, -config.reward_clip, config.reward_clip))
                replay_buffer.add(state, action, stored_reward, next_state, bool(terminated))
                state = next_state
                episode_reward += float(reward)
                episode_length += 1
                global_step += 1
                max_q_values.append(max_q)
                action_counts[action] += 1

                if (
                    global_step >= config.learning_starts
                    and global_step % config.train_frequency == 0
                    and len(replay_buffer) >= config.batch_size
                ):
                    online_network.train()
                    metrics.append(
                        optimize_batch(
                            online_network,
                            target_network,
                            optimizer,
                            replay_buffer,
                            config,
                            device,
                        )
                    )
                    if config.target_tau is not None:
                        soft_update_target(online_network, target_network, config.target_tau)
                if (
                    config.target_tau is None
                    and global_step % config.target_update_frequency == 0
                ):
                    target_network.load_state_dict(online_network.state_dict())

            row: dict[str, float | int] = {
                "episode": episode,
                "reward": episode_reward,
                "length": episode_length,
                "epsilon": epsilon_at_step(config, global_step),
                "mean_max_q": float(np.mean(max_q_values)),
                "buffer_size": len(replay_buffer),
                "global_step": global_step,
            }
            for name in (
                "loss",
                "mean_abs_td_error",
                "online_target_gap",
                "gradient_norm_before_clipping",
            ):
                row[name] = float(np.mean([metric[name] for metric in metrics])) if metrics else np.nan
            records.append(row)
            action_rows.append(action_counts)

            recent_rewards = [float(record["reward"]) for record in records[-100:]]
            recent_mean = float(np.mean(recent_rewards))
            if len(records) >= 100 and recent_mean > best_moving_average:
                best_moving_average = recent_mean
                save_checkpoint(
                    run_dir / "best_checkpoint.pt",
                    online_network,
                    target_network,
                    optimizer,
                    config,
                    episode,
                    global_step,
                    best_moving_average,
                )

            if episode == 1 or episode % 10 == 0:
                print(
                    f"episode={episode} reward={episode_reward:.2f} length={episode_length} "
                    f"mean_100={recent_mean:.2f} epsilon={row['epsilon']:.3f} "
                    f"steps={global_step}",
                    flush=True,
                )

            should_checkpoint = episode % config.checkpoint_every == 0 or episode == config.episodes
            reached_stop = (
                config.stop_score is not None
                and episode >= config.minimum_episodes_before_stop
                and len(records) >= 100
                and recent_mean >= config.stop_score
            )
            if should_checkpoint or reached_stop:
                history = pd.DataFrame.from_records(records)
                history.to_csv(history_path, index=False)
                action_matrix = np.stack(action_rows)
                pd.DataFrame(
                    action_matrix,
                    columns=[f"action_{index}" for index in range(config.n_actions)],
                ).to_csv(actions_path, index=False)
                save_checkpoint(
                    checkpoint_path,
                    online_network,
                    target_network,
                    optimizer,
                    config,
                    episode,
                    global_step,
                    best_moving_average,
                )
                plot_training_history(history, run_dir)
                plot_action_frequency(action_matrix, run_dir)
            if reached_stop:
                print(f"stopping after reaching mean_100={recent_mean:.2f}", flush=True)
                break
    finally:
        environment.close()

    return pd.DataFrame.from_records(records)


def load_run(
    run_dir: Path,
    device_name: str = "auto",
    checkpoint_name: str = "best_checkpoint.pt",
) -> tuple[DQNConfig, nn.Module, torch.device]:
    config_data = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    config_data.pop("resolved_gamma", None)
    config = DQNConfig(**config_data)
    device = resolve_device(device_name)
    network = build_network(config).to(device)
    checkpoint_path = run_dir / checkpoint_name
    if not checkpoint_path.exists():
        checkpoint_path = run_dir / "checkpoint.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    network.load_state_dict(checkpoint["online_state_dict"])
    network.eval()
    return config, network, device


def evaluate(
    run_dir: Path,
    episodes: int = 500,
    epsilon: float = 0.0,
    device_name: str = "auto",
    checkpoint_name: str = "best_checkpoint.pt",
) -> pd.DataFrame:
    config, network, device = load_run(run_dir, device_name, checkpoint_name)
    environment = make_environment(config)
    stack = FrameStack(config.frame_stack) if config.env == "mspacman" else None
    generator = np.random.default_rng(config.seed + 200_000)
    rows: list[dict[str, float | int]] = []
    try:
        for episode in range(1, episodes + 1):
            state = reset_state(environment, config, config.seed + 100_000 + episode, stack)
            terminated = truncated = False
            reward_sum = 0.0
            length = 0
            while not (terminated or truncated):
                action, _, _ = select_action(
                    network, state, epsilon, config.n_actions, generator, device
                )
                observation, reward, terminated, truncated, _ = environment.step(action)
                state = next_state_from_observation(observation, config, stack)
                reward_sum += float(reward)
                length += 1
            rows.append({"episode": episode, "reward": reward_sum, "length": length})
            if episode == 1 or episode % 25 == 0:
                print(f"evaluation_episode={episode} reward={reward_sum:.2f}", flush=True)
    finally:
        environment.close()

    frame = pd.DataFrame(rows)
    frame.to_csv(run_dir / "evaluation_greedy.csv", index=False)
    rewards = frame["reward"].to_numpy(dtype=np.float64)
    summary = {
        "episodes": episodes,
        "policy": "greedy" if epsilon == 0 else f"epsilon={epsilon:g}",
        "checkpoint": checkpoint_name,
        "mean_reward": float(np.mean(rewards)),
        "standard_deviation": float(np.std(rewards, ddof=0)),
        "minimum_reward": float(np.min(rewards)),
        "maximum_reward": float(np.max(rewards)),
        "mean_episode_length": float(frame["length"].mean()),
    }
    (run_dir / "evaluation_greedy_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.hist(rewards, bins="auto", color="#3C91E6", edgecolor="white")
    ax.axvline(
        summary["mean_reward"], color="#C1121F", linestyle="--", linewidth=2,
        label=f"Mean = {summary['mean_reward']:.2f}",
    )
    ax.set(
        xlabel="Episode reward",
        ylabel="Count",
        title=f"Distribution of rewards over {episodes} evaluation episodes",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "evaluation_greedy_histogram.png", dpi=200)
    fig.savefig(run_dir / "evaluation_greedy_histogram.pdf")
    plt.close(fig)
    print(json.dumps(summary, indent=2))
    return frame
