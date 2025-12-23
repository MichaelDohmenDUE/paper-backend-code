from collections import namedtuple
from copy import deepcopy
from typing import Tuple

import torch
from networkx.algorithms.tree import from_nested_tuple
from sympy.plotting import plot3d
from torch import nn
import random
import gymnasium as gym
import numpy as np
from backend.Utils.src.utils import synchronize
import torch.nn.functional as F
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


Transition = namedtuple("transition", ["state", "action", "reward", "next_state", "done"])


class DataCollectionProcessor():
    def __init__(self, policy: nn.Module, env: gym.Env, buffer: ReplayBuffer):
        self.policy = policy
        self.env = env
        self.buffer = buffer
        self.state, _ = env.reset()
        self.done = False

    def run(self) -> Transition:
        q_values = behavior_net(torch.tensor(self.state, dtype=torch.float32))
        action = epsilon_greedy(q_values=q_values, epsilon=epsilon)
        next_state, reward, terminated, truncated, info = env.step(action.item())
        done = terminated or truncated

        # transition = (state, action.item(), reward, next_state, done)
        transition = Transition(self.state, action.item(), reward, next_state, done)
        buffer.append(transition)
        self.state = next_state
        if done:
            self.state, _ = env.reset()
            self.done = False

        return transition

class TrainProcessor():
    def __init__(self, buffer: ReplayBuffer, behavior_net: nn.Module, target_net: nn.Module, optimizer: torch.optim.Optimizer):
        self.buffer = buffer
        self.behavior_net = behavior_net
        self.target_net = target_net
        self.optimizer = optimizer

    def run(self):
        if len(buffer) < batch_size:
            return

        transitions = buffer.sample()
        states, actions, rewards, next_states, dones = zip(*transitions)


        states_tensor = torch.tensor(np.array(states), dtype=torch.float32)  # batch_size x 4
        actions_tensor = torch.tensor(actions, dtype=torch.int64).unsqueeze(-1)  # batch_size x 1
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32).unsqueeze(-1)
        next_states_tensor = torch.tensor(np.array(next_states), dtype=torch.float32)
        dones_tensor = torch.tensor(dones, dtype=torch.float32).unsqueeze(-1)

        qsa_behavior = behavior_net(states_tensor).gather(1, actions_tensor)  # ^y


        qs_target = target_net(next_states_tensor)  # batch_size x action_dim
        qsa_target = torch.max(qs_target, dim=1).values.unsqueeze(-1).detach()
        target = rewards_tensor + gamma * qsa_target * (1.0 - dones_tensor)  # [~dones_tensor]  # y
        target = target.detach()

        # ToDo: How to model dependent steps without forwarding anything
        optimizer.zero_grad()
        loss = F.mse_loss(qsa_behavior, target)
        loss.backward()
        optimizer.step()

        return loss.item()

class SyncProcessor():
    def __init__(self, from_net : nn.Module, to_net : nn.Module, tau = 1.0):
        self.from_net = from_net
        self.to_net = to_net
        self.tau = tau

    def run(self):
        if total_steps % sync_freq != 0:
            return
        if self.tau == 1.0:
            self.hard_sync()
        else:
            self.soft_sync()

    def hard_sync(self):
        self.to_net.load_state_dict(self.from_net.state_dict())

    def soft_sync(self):
        for from_param, to_param in zip(self.from_net.parameters(), self.to_net.parameters()):
            to_param.copy_(self.tau * from_param + (1.0 - self.tau) * to_param)

if __name__ == '__main__':

    # initialization
    # ToDo: Constants as global variables or member variables
    lr = 1e-3
    num_episodes = 1000
    epsilon = 0.2
    env_name = "CartPole-v1"
    sync_freq = 40
    hidden_size = 32
    batch_size = 64
    max_buffer_size = 10000
    TAU = 1.0
    gamma = 0.99

    env = gym.make(env_name)
    obs_size, action_size, max_action = get_env_specs(env)

    behavior_net = nn.Sequential(nn.Linear(obs_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, action_size))
    optimizer = torch.optim.Adam(behavior_net.parameters(), lr)

    target_net = deepcopy(behavior_net)

    buffer = ReplayBuffer(max_buffer_size, batch_size)

    collector = DataCollectionProcessor(behavior_net, env, buffer)
    train_process = TrainProcessor(buffer, behavior_net, target_net, optimizer)
    sync_process = SyncProcessor(behavior_net, target_net)
    # Outer training loop
    total_steps = 0
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False

        episode_rewards = 0
        while not done:
            transition = collector.run()
            train_process.run()
            sync_process.run()

            total_steps += 1
            done = transition.done
            episode_rewards += transition.reward
            # synchronization
        if episode % 10 == 0:
            print(f"Episode [{episode}] {episode_rewards}")
