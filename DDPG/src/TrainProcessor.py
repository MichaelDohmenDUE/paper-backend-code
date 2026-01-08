import torch
import torch.nn.functional as F
from torch import nn

from backend.Utils.src.ReplayBuffer import ReplayBuffer


class TrainProcess:
    def __init__(self, buffer: ReplayBuffer, actor: nn.Module, actor_target: nn.Module, critic: nn.Module,
                 critic_target: nn.Module, actor_optimizer: torch.optim.Optimizer,
                 critic_optimizer: torch.optim.Optimizer, gamma, device):
        self.buffer = buffer
        self.actor = actor.to(device)
        self.actor_target = actor_target.to(device)
        self.critic = critic.to(device)
        self.critic_target = critic_target.to(device)
        self.actor_opt = actor_optimizer
        self.critic_opt = critic_optimizer
        self.gamma = gamma
        self.device = device

    def run(self):
        if len(self.buffer) < self.buffer.batch_size:
            return None, None

        batch = self.buffer.sample_batch()

        states = batch["state"].to(self.device)
        actions = batch["action"].to(self.device)
        rewards = batch["reward"].to(self.device)
        next_states = batch["next_state"].to(self.device)
        dones = batch["done"].to(self.device)

        # Critic update
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            target = rewards + self.gamma * target_q * (1.0 - dones)
        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, target)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()
        # Actor update
        self.actor_opt.zero_grad()
        actor_loss = -self.critic(states, self.actor(states)).mean()
        actor_loss.backward()
        self.actor_opt.step()
        return actor_loss, critic_loss
