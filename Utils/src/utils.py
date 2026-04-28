import random

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn


def setting_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def get_module_grad_stats(module: nn.Module):
    total_squared_norm = 0.0
    n_params = 0
    for p in module.parameters():
        if p.grad is not None:
            total_squared_norm += p.grad.detach().pow(2).sum().item()
            n_params += p.numel()
    rms_grad = (total_squared_norm / n_params) ** 0.5 if n_params > 0 else 0.0
    return rms_grad, n_params

def gae(gamma: float, lambda_: float, deltas, dones):
    advantages = torch.zeros_like(deltas)
    advantage = 0.0
    for t in reversed(range(len(deltas))):
        advantage = deltas[t] + gamma * lambda_ * advantage * (1.0 - dones[t].float())
        advantages[t] = advantage
    return advantages


def compute_gae(buffer, gamma: float, lam: float):
    states = np.array([t.state for t in buffer.buffer])
    actions = np.array([t.action for t in buffer.buffer])
    logps = np.array([t.logp for t in buffer.buffer], dtype=np.float32)
    rewards = np.array([t.reward for t in buffer.buffer], dtype=np.float32)
    dones = np.array([t.done for t in buffer.buffer], dtype=np.float32)
    values = np.array([t.value for t in buffer.buffer], dtype=np.float32)

    advantages = np.zeros_like(rewards)
    last_gae_lam = 0.0

    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = buffer.buffer[t].bootstrap_value
            next_non_terminal = 1.0 - dones[t]
        else:
            next_value = values[t + 1]
            next_non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]

        advantages[t] = last_gae_lam = delta + gamma * lam * next_non_terminal * last_gae_lam

    returns = advantages + values

    return states, actions, logps, advantages, returns


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
    masked_next_state_values = next_state_values * (1. - dones.reshape(-1, 1))

    td_residuals = rewards.reshape(-1, 1) + gamma * masked_next_state_values - state_values
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
