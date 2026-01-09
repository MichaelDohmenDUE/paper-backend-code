import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim, nn

from backend.CommonModels.src.Actor import Actor
from backend.CommonModels.src.Critic import Critic
from backend.Utils.src.ReplayBuffer import ReplayBuffer
from backend.Utils.src.utils import synchronize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TrainProcessor:
    """
    Twin Delayed Deep Deterministic Policy Gradient (TD3)
    Paper: https://arxiv.org/abs/1802.09477
    """

    def __init__(self, actor: nn.Module ,replay_buffer: ReplayBuffer, state_size: int, action_size: int, hidden_size: int, max_action: float, learning_rate: float,
                 tau: float, noise_clip: float, policy_noise, start_timesteps=25000, synchro_frequency: int = 2, discount_factor: int = 0.99):
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.max_action = max_action
        self.learning_rate = learning_rate
        self.tau = tau
        self.noise_clip = noise_clip
        self.policy_noise = policy_noise
        self.syncro_frequency = synchro_frequency
        self.discount_factor = discount_factor
        self.iteration = 0
        self.global_timestep = 0
        self.start_timesteps = start_timesteps
        self.actor = actor.to(device)
        self.replay_buffer = replay_buffer
        self.device = device

        self.actor_target = copy.deepcopy(self.actor)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=learning_rate)

        self.critic_1 = Critic(state_size, action_size, hidden_size).to(device)
        self.critic_2 = Critic(state_size, action_size, hidden_size).to(device)
        self.critic_target_1 = copy.deepcopy(self.critic_1)
        self.critic_target_2 = copy.deepcopy(self.critic_2)

        self.optimizer_critic_1 = optim.Adam(self.critic_1.parameters(), lr=learning_rate)
        self.optimizer_critic_2 = optim.Adam(self.critic_2.parameters(), lr=learning_rate)

    def run(self):
        if self.global_timestep >= self.start_timesteps:
            self.train()
        self.global_timestep += 1

    def train(self):
        self.iteration += 1

        batch = self.replay_buffer.sample_batch()
        state = batch["state"].to(self.device)
        action = batch["action"].to(self.device)
        reward = batch["reward"].to(self.device)
        next_state = batch["next_state"].to(self.device)
        done = batch["done"].to(self.device)
        valid_transition = 1.0 - done

        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

            target_Q1 = self.critic_target_1(next_state, next_action)
            target_Q2 = self.critic_target_2(next_state, next_action)

            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + valid_transition * self.discount_factor * target_Q

        current_Q1 = self.critic_1(state, action)
        current_Q2 = self.critic_2(state, action)

        critic_loss_1 = F.mse_loss(current_Q1, target_Q)
        critic_loss_2 = F.mse_loss(current_Q2, target_Q)

        self.optimizer_critic_1.zero_grad()
        critic_loss_1.backward()
        self.optimizer_critic_1.step()

        self.optimizer_critic_2.zero_grad()
        critic_loss_2.backward()
        self.optimizer_critic_2.step()

        if self.iteration % self.syncro_frequency == 0:
            actor_loss = -self.critic_1(state, self.actor(state)).mean()

            self.optimizer_actor.zero_grad()
            actor_loss.backward()
            self.optimizer_actor.step()

            synchronize(self.critic_1, self.critic_target_1, self.tau)
            synchronize(self.critic_2, self.critic_target_2, self.tau)
            synchronize(self.actor, self.actor_target, self.tau)
