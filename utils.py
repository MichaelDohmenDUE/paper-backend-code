import numpy as np
from numpy.typing import NDArray
import torch
from torch import nn
from typing import Iterable, Any
import random
from collections import deque

def gae(gamma: float, lambda_: float, deltas: NDArray[np.float64], dones: NDArray[np.bool_]) -> NDArray[np.float64]:
    advantages = np.empty_like(deltas)
    advantage = 0.0
    for t in reversed(range(len(deltas))):
        advantage = deltas[t] + gamma * lambda_ * advantage * (1 - dones[t])
        advantages[t] = advantage
    return advantages


def discounted_cumulative_reward(gamma: float, rewards: NDArray[np.float64], dones: NDArray[np.bool_]) -> NDArray[
    np.float64]:
    total_rewards = np.empty_like(rewards)
    G = 0.0
    for t in reversed(range(len(rewards))):
        G = rewards[t] + gamma * G * (1 - dones[t])
        total_rewards[t] = G
    return total_rewards


def temporal_difference_residuals(gamma: float, rewards: NDArray[np.float64],
                                  state_values: NDArray[np.float64],
                                  next_state_values: NDArray[np.float64],
                                  dones: NDArray[np.bool_]) -> NDArray[np.float64]:

    masked_next_state_values = next_state_values * (1. - dones.reshape(-1,1))

    td_residuals = rewards.reshape(-1,1) + gamma * masked_next_state_values - state_values
    return td_residuals


def synchronize(from_network: nn.Module, to_network: nn.Module, tau: float) -> None:
    """
    Synchronize parameters from one network to another using soft or hard updates.

    Args:
        from_network (nn.Module): The source network (e.g., behavior network).
        to_network (nn.Module): The target network to be updated.
        tau (float): Interpolation factor for the update.
            - tau = 1.0 → hard update.
            - 0 < tau < 1.0 → soft update.
            - tau = 0.0 → no update.

    Returns:
        None

    Raises:
        AssertionError: If `tau` is not within [0.0, 1.0]
    """

    assert 0.0 <= tau <= 1.0, f"tau must be in [0, 1], got {tau}"

    with torch.no_grad():
        if tau == 1.0:
            to_network.load_state_dict(from_network.state_dict())
        else:
            for from_param, to_param in zip(from_network.parameters(), to_network.parameters()):
                to_param.copy_(tau * from_param + (1.0 - tau) * to_param)


class ReplayBuffer:
    def __init__(self, buffer_size: int = 10_000):
        self.buffer: deque[Any] = deque(maxlen=buffer_size)

    def __len__(self) -> int:
        return len(self.buffer)

    def __str__(self) -> str:
        return f"{list(self.buffer)}"

    def append(self, x: Any) -> None:
        self.buffer.append(x)

    def extend(self, iterable: Iterable[Any]) -> None:
        self.buffer.extend(iterable)

    def sample(self, batch_size: int) -> list[Any]:
        return random.sample(self.buffer, batch_size)

    def choice(self, indices : list[Any]) -> list[Any]:
        indexed_list = [self.buffer[idx] for idx in indices]
        return indexed_list

