from collections import namedtuple
from copy import deepcopy
from typing import Tuple

import torch
from torch import nn
import random
import gymnasium as gym
import numpy as np
import torch.nn.functional as F

from backend.Utils.src.BatchTransitioner import TransitionSpec, TransitionFactory, TransitionBatch
from backend.Utils.src.ReplayBuffer import ReplayBuffer

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


class DataCollectionProcessor:
    def __init__(self, policy: nn.Module, env: gym.Env, buffer: ReplayBuffer, action_selector: EpsilonGreedyPolicy, transition_factory: TransitionFactory):
        self.policy = policy
        self.env = env
        self.buffer = buffer
        self.state, _ = env.reset()
        self.done = False
        self.action_selector = action_selector
        self.transition_factory = transition_factory
        # Logging
        self.episode_count = 0
        self.episode_reward = 0
        self.total_steps = 0

    def run(self) -> None:
        if self.done:
            if self.episode_count % 10 == 0:
                print(f"Episode [{self.episode_count}] {self.episode_reward}")

            self.episode_count += 1
            self.episode_reward = 0.0

            self.state, _ = self.env.reset()
            self.done = False
        with torch.no_grad():
            q_values = self.policy(torch.as_tensor(self.state, dtype=torch.float32))
        action = self.action_selector.select_action(q_values=q_values)
        next_state, reward, terminated, truncated, info = self.env.step(action.item())
        self.done = terminated or truncated

        transition = self.transition_factory.create( state=self.state, action=action.item(), reward=reward, next_state=next_state, done=self.done )
        self.buffer.append(transition)
        self.episode_reward += reward

        self.state = next_state

        self.total_steps += 1

class TrainProcessor:
    def __init__(self, buffer: ReplayBuffer, behavior_net: nn.Module, target_net: nn.Module,
                 optimizer: torch.optim.Optimizer, gamma: float):
        self.buffer = buffer
        self.behavior_net = behavior_net
        self.target_net = target_net
        self.optimizer = optimizer
        self.gamma = gamma

    def run(self):
        if len(self.buffer) < self.buffer.batch_size:
            return

        batch = self.buffer.sample_batch()

        states_tensor      = batch["state"]
        actions_tensor     = batch["action"]
        rewards_tensor     = batch["reward"]
        next_states_tensor = batch["next_state"]
        dones_tensor       = batch["done"]

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
    max_steps = 100000
    spec = TransitionSpec(["state", "action", "reward", "next_state", "done"])
    factory = TransitionFactory(spec)

    env = gym.make(env_name)
    obs_size, action_size, max_action = get_env_specs(env)

    behavior_net = nn.Sequential(nn.Linear(obs_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, action_size))
    optimizer = torch.optim.Adam(behavior_net.parameters(), lr)

    target_net = deepcopy(behavior_net)

    buffer = ReplayBuffer(spec, max_buffer_size, batch_size)
    collector = DataCollectionProcessor(behavior_net, env, buffer, EpsilonGreedyPolicy(epsilon), factory)
    train_process = TrainProcessor(buffer, behavior_net, target_net, optimizer, gamma)
    sync_process = SyncProcessor(behavior_net, target_net, tau, sync_freq)


    for step in range(max_steps):
        collector.run()
        train_process.run()
        sync_process.run()


if __name__ == '__main__':
    main()
