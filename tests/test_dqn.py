import numpy as np
import pytest
import torch
from torch import nn

from mlds490_hw2.dqn import (
    DQNConfig,
    FrameStack,
    ReplayBuffer,
    build_network,
    compute_double_dqn_targets,
    epsilon_at_step,
    moving_average,
    preprocess_mspacman,
    soft_update_target,
)


def test_assignment_preprocessing_shape_type_and_range() -> None:
    frame = np.zeros((210, 160, 3), dtype=np.uint8)
    frame[1, 0] = [210, 164, 74]
    frame[3, 2] = [255, 255, 255]
    processed = preprocess_mspacman(frame)
    assert processed.shape == (88, 80)
    assert processed.dtype == np.int8
    assert processed.min() >= -128
    assert processed.max() <= 127
    assert processed[0, 0] == -128


def test_preprocessing_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError):
        preprocess_mspacman(np.zeros((88, 80), dtype=np.uint8))


def test_frame_stack_resets_without_cross_episode_leakage() -> None:
    stack = FrameStack(4)
    first = np.zeros((210, 160, 3), dtype=np.uint8)
    second = np.full((210, 160, 3), 255, dtype=np.uint8)
    first_state = stack.reset(first)
    stack.append(second)
    reset_state = stack.reset(first)
    assert first_state.shape == (4, 88, 80)
    np.testing.assert_array_equal(reset_state, first_state)


def test_replay_buffer_wraps_and_samples_valid_data() -> None:
    buffer = ReplayBuffer(3, (2,), np.dtype(np.float32), seed=1)
    for index in range(5):
        state = np.array([index, index + 1], dtype=np.float32)
        buffer.add(state, index % 2, float(index), state + 1, index == 4)
    assert len(buffer) == 3
    states, actions, rewards, next_states, terminated = buffer.sample(3)
    assert states.shape == (3, 2)
    assert set(rewards.tolist()) == {2.0, 3.0, 4.0}
    assert set(actions.tolist()).issubset({0, 1})
    np.testing.assert_allclose(next_states, states + 1)
    assert terminated.sum() == 1


def test_replay_buffer_rejects_oversampling() -> None:
    buffer = ReplayBuffer(4, (1,), np.dtype(np.float32), seed=1)
    buffer.add(np.array([0], dtype=np.float32), 0, 0.0, np.array([1]), False)
    with pytest.raises(ValueError):
        buffer.sample(2)


def test_epsilon_schedule_has_expected_endpoints() -> None:
    config = DQNConfig(
        env="cartpole",
        episodes=1,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay_steps=100,
    )
    assert epsilon_at_step(config, 0) == pytest.approx(1.0)
    assert epsilon_at_step(config, 50) == pytest.approx(0.55)
    assert epsilon_at_step(config, 100) == pytest.approx(0.1)
    assert epsilon_at_step(config, 1000) == pytest.approx(0.1)


class FixedNetwork(nn.Module):
    def __init__(self, values: list[float]) -> None:
        super().__init__()
        self.register_buffer("values", torch.tensor(values, dtype=torch.float32))

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.values.repeat(len(states), 1)


def test_double_dqn_targets_do_not_bootstrap_terminal_transitions() -> None:
    online = FixedNetwork([1.0, 3.0])
    target = FixedNetwork([10.0, 20.0])
    next_states = torch.zeros(2, 4)
    rewards = torch.tensor([2.0, 2.0])
    terminated = torch.tensor([0.0, 1.0])
    targets = compute_double_dqn_targets(
        online, target, next_states, rewards, terminated, gamma=0.5
    )
    torch.testing.assert_close(targets, torch.tensor([12.0, 2.0]))


def test_soft_target_update_moves_toward_online_network() -> None:
    online = nn.Linear(1, 1, bias=False)
    target = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        online.weight.fill_(2.0)
        target.weight.fill_(0.0)
    soft_update_target(online, target, tau=0.25)
    torch.testing.assert_close(target.weight, torch.tensor([[0.5]]))


def test_network_output_shapes() -> None:
    cartpole = build_network(DQNConfig(env="cartpole", episodes=1))
    mspacman = build_network(DQNConfig(env="mspacman", episodes=1))
    assert cartpole(torch.zeros(3, 4)).shape == (3, 2)
    assert mspacman(torch.zeros(3, 4, 88, 80)).shape == (3, 9)


def test_moving_average_keeps_episode_alignment() -> None:
    result = moving_average(np.arange(1, 6, dtype=float), window=3)
    assert np.isnan(result[:2]).all()
    np.testing.assert_allclose(result[2:], [2.0, 3.0, 4.0])


def test_configuration_uses_assignment_discounts() -> None:
    assert DQNConfig(env="cartpole", episodes=1).resolved_gamma == 0.95
    assert DQNConfig(env="mspacman", episodes=1).resolved_gamma == 0.99
