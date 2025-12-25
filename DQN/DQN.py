from collections import namedtuple
from copy import deepcopy
from typing import Tuple

import torch
from torch import nn
import random
import gymnasium as gym
import numpy as np
import torch.nn.functional as F

from backend.Utils.src.ReplayBuffer import ReplayBuffer

def preprocess(data):
    if isinstance(data[0], int | np.int64):
        data_tensor = torch.tensor(data, dtype=torch.int64).unsqueeze(-1)
    elif isinstance(data[0], bool | float):
        data_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(-1)
    elif isinstance(data[0], np.ndarray):
        data_tensor = torch.tensor(np.array(data), dtype=torch.float32)
    else:
        raise ValueError("Unknown data type")
    return data_tensor

def epsilon_greedy(q_values: torch.Tensor, epsilon: float) -> torch.Tensor:
    """ Epsilon-greedy policy, returns random action if random number is < epsilon, else greedy action
        Randomly samples from max actions if there is a tie.
     """
    actions = torch.arange(len(q_values))
    max_q_value = torch.max(q_values)
    max_idx = (q_values == max_q_value).to(torch.int64)
    greedy_action = random.choice(actions[max_idx == 1])

    return random.choice(actions) if random.random() < epsilon else greedy_action


def get_env_specs(env: gym.Env) -> Tuple[int, int, float]:
    observation_dim = env.observation_space.shape[0]

    if isinstance(env.action_space, gym.spaces.Discrete):
        action_dim = env.action_space.n
        max_action = None
    else:
        action_dim = env.action_space.shape[0]
        max_action = float(env.action_space.high[0])

    return observation_dim, action_dim, max_action

class EpsilonGreedyPolicy:
    def __init__(self, epsilon: float):
        self.epsilon = epsilon

    def select_action(self, q_values: torch.Tensor) -> torch.Tensor:
        """ Epsilon-greedy policy, returns random action if random number is < epsilon, else greedy action
                Randomly samples from max actions if there is a tie.
             """
        actions = torch.arange(len(q_values))
        max_q_value = torch.max(q_values)
        max_idx = (q_values == max_q_value).to(torch.int64)
        greedy_action = random.choice(actions[max_idx == 1])

        return random.choice(actions) if random.random() < self.epsilon else greedy_action


Transition = namedtuple("transition", ["state", "action", "reward", "next_state", "done"])


class DataCollectionProcessor:
    def __init__(self, policy: nn.Module, env: gym.Env, buffer: ReplayBuffer, action_selector: EpsilonGreedyPolicy):
        self.policy = policy
        self.env = env
        self.buffer = buffer
        self.state, _ = env.reset()
        self.done = False
        self.action_selector = action_selector
        # Logging
        self.episode_count = 0
        self.episode_reward = 0

    def run(self) -> None:
        """
        Thoughts: An episode is only relevant for logging. When an episode terminates, the done signal is relevant for
        the update. The done signal is also relevant for resetting the environment. So we are basically only interested
        in whether we have to reset or not and wheter we have a next state or not. So let's try to do the steps without
        looping over episodes. We can still log, when an episode terminates, and also count the number of episodes.
        Returns:

        """
        if self.done:
            if self.episode_count % 10 == 0:
                print(f"Episode [{self.episode_count}] {self.episode_reward}")

            self.episode_count += 1
            self.episode_reward = 0.0

            self.state, _ = self.env.reset()
            self.done = False

        q_values = self.policy(torch.tensor(self.state, dtype=torch.float32))
        action = self.action_selector.select_action(q_values=q_values)
        next_state, reward, terminated, truncated, info = self.env.step(action.item())
        self.done = terminated or truncated

        transition = Transition(self.state, action.item(), reward, next_state, self.done)
        self.buffer.append(transition)
        self.episode_reward += reward

        self.state = next_state

class TrainProcessor:
    def __init__(self, buffer: ReplayBuffer, behavior_net: nn.Module, target_net: nn.Module,
                 optimizer: torch.optim.Optimizer, gamma: float):
        self.buffer = buffer
        self.behavior_net = behavior_net
        self.target_net = target_net
        self.optimizer = optimizer
        self.gamma = gamma

    def run(self) -> None:
        if len(self.buffer) < self.buffer.batch_size:
            return

        transitions = self.buffer.sample()
        states, actions, rewards, next_states, dones = zip(*transitions)

        states_tensor = preprocess(states)
        actions_tensor = preprocess(actions)  # batch_size x 1
        rewards_tensor = preprocess(rewards)
        next_states_tensor = preprocess(next_states)
        dones_tensor = preprocess(dones)

        qsa_behavior = self.behavior_net(states_tensor).gather(1, actions_tensor)  # ^y

        qs_target = self.target_net(next_states_tensor)  # batch_size x action_dim
        qsa_target = torch.max(qs_target, dim=1).values.unsqueeze(-1).detach()
        target = rewards_tensor + self.gamma * qsa_target * (1.0 - dones_tensor)
        target = target.detach()

        # ToDo: How to model dependent steps without forwarding anything
        self.optimizer.zero_grad()
        loss = F.mse_loss(qsa_behavior, target)
        loss.backward()
        self.optimizer.step()


class SyncProcessor:
    def __init__(self, from_net: nn.Module, to_net: nn.Module, tau: float, sync_freq: int):
        self.from_net = from_net
        self.to_net = to_net
        self.tau = tau
        # Each process runs sequentially and can
        self.counter = 0
        self.sync_freq = sync_freq

    def run(self) -> None:
        if self.counter % self.sync_freq == 0:
            if self.tau == 1.0:
                self.hard_sync()
            else:
                self.soft_sync()
        self.counter +=1

    def hard_sync(self):
        self.to_net.load_state_dict(self.from_net.state_dict())

    def soft_sync(self):
        for from_param, to_param in zip(self.from_net.parameters(), self.to_net.parameters()):
            to_param.copy_(self.tau * from_param + (1.0 - self.tau) * to_param)


def main():
    # initialization
    # ToDo: Constants as global variables or member variables
    lr = 1e-3
    epsilon = 0.2
    env_name = "CartPole-v1"
    sync_freq = 40
    hidden_size = 32
    batch_size = 64
    max_buffer_size = 10000
    tau = 1.0
    gamma = 0.99
    max_steps = 10000

    env = gym.make(env_name)
    obs_size, action_size, max_action = get_env_specs(env)

    behavior_net = nn.Sequential(nn.Linear(obs_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, action_size))
    optimizer = torch.optim.Adam(behavior_net.parameters(), lr)

    target_net = deepcopy(behavior_net)

    buffer = ReplayBuffer(max_buffer_size, batch_size)

    collector = DataCollectionProcessor(behavior_net, env, buffer, EpsilonGreedyPolicy(epsilon))
    train_process = TrainProcessor(buffer, behavior_net, target_net, optimizer, gamma)
    sync_process = SyncProcessor(behavior_net, target_net, tau, sync_freq)


    for step in range(max_steps):
        collector.run()
        train_process.run()
        sync_process.run()


if __name__ == '__main__':
    main()
