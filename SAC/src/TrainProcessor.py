import torch
from torch import nn
import torch.nn.functional as F
from backend.Utils.src.ReplayBuffer import ReplayBuffer


class TrainProcessor:
    def __init__(self, buffer: ReplayBuffer, actor: nn.Module, critic_1: nn.Module,
                 critic_target_1: nn.Module, critic_2: nn.Module, critic_target_2: nn.Module, actor_optimizer: torch.optim.Optimizer,
                 critic_optimizer_1: torch.optim.Optimizer,critic_optimizer_2: torch.optim.Optimizer, gamma, device):
        self.buffer = buffer
        self.actor = actor.to(device)
        self.critic_1 = critic_1.to(device)
        self.critic_target_1 = critic_target_1.to(device)
        self.critic_2 = critic_2.to(device)
        self.critic_target_2 = critic_target_2.to(device)
        self.actor_opt = actor_optimizer
        self.critic_opt_1 = critic_optimizer_1
        self.critic_opt_2 = critic_optimizer_2
        self.gamma = gamma
        self.device = device

    def run(self):
        if len(self.buffer) < self.buffer.batch_size:
            return None, None

        batch = self.buffer.sample_batch()

        states      = batch["state"].to(self.device)
        actions     = batch["action"].to(self.device)
        rewards     = batch["reward"].to(self.device)
        next_states = batch["next_state"].to(self.device)
        dones       = batch["done"].to(self.device)

        # Critic update
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target_1(next_states, next_actions)
            target = rewards + self.gamma * target_q * (1.0 - dones)
        current_q = self.critic_1(states, actions)
        critic_loss = F.mse_loss(current_q, target)

        self.critic_opt_1.zero_grad()
        critic_loss.backward()
        self.critic_opt_1.step()
        # Actor update
        self.actor_opt.zero_grad()
        actor_loss = -self.critic_1(states, self.actor(states)).mean()
        actor_loss.backward()
        self.actor_opt.step()
        return actor_loss, critic_loss
