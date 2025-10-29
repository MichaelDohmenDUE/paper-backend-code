import random

import torch

import numpy as np
from backend.Utils.src.utils import synchronize
import torch.nn.functional as F

class DDQNTrainer:
    def __init__(
        self,
        env_handler,
        behavior_policy,
        target_policy,
        optimizer,
        buffer,
        gamma=0.99,
        batch_size=64,
        update_freq=40,
        epsilon=0.2,
        device="cpu"
    ):
        self.env_handler = env_handler
        self.behavior_policy = behavior_policy
        self.target_policy = target_policy
        self.optimizer = optimizer
        self.buffer = buffer
        self.gamma = gamma
        self.batch_size = batch_size
        self.update_freq = update_freq
        self.epsilon = epsilon
        self.device = device
        self.total_steps = 0

    def train(self):
            if len(self.buffer) > self.batch_size:
                self._optimize()
                self.total_steps += 1
                if self.total_steps % self.update_freq == 0:
                    synchronize(self.behavior_policy, self.target_policy, tau=1.0)

    def _optimize(self):
        transitions = self.buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)

        states_tensor = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions_tensor = torch.tensor(actions, dtype=torch.int64).unsqueeze(-1).to(self.device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32).unsqueeze(-1).to(self.device)
        next_states_tensor = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        dones_tensor = torch.tensor(dones, dtype=torch.float32).unsqueeze(-1).to(self.device)

        qsa_behavior = self.behavior_policy(states_tensor).gather(1, actions_tensor)

        next_qs_behavior = self.behavior_policy(next_states_tensor)
        next_actions = torch.argmax(next_qs_behavior, dim=1, keepdim=True)

        next_qs_target = self.target_policy(next_states_tensor)
        qsa_target = next_qs_target.gather(1, next_actions).detach()

        target = rewards_tensor + self.gamma * qsa_target * (1.0 - dones_tensor)

        loss = F.mse_loss(qsa_behavior, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()