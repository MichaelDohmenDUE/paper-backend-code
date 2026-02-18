import torch
import torch.nn.functional as F
from torch import nn

from backend.Utils.src.ReplayBuffer import ReplayBuffer


class TrainProcessor:
    def __init__(self, buffer: ReplayBuffer, actor: nn.Module, critic_1: nn.Module,
                 critic_target_1: nn.Module, critic_2: nn.Module, critic_target_2: nn.Module,
                 actor_optimizer: torch.optim.Optimizer,
                 critic_optimizer_1: torch.optim.Optimizer, critic_optimizer_2: torch.optim.Optimizer,
                 log_alpha, alpha_optimizer, target_entropy, gamma, device):
        self.target_entropy = target_entropy
        self.log_alpha = log_alpha
        self.alpha_optimizer = alpha_optimizer
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
            return None, None, None, None

        current_alpha = self.log_alpha.exp()

        batch = self.buffer.sample_batch()

        states = batch["state"].to(self.device)
        actions = batch["action"].to(self.device)
        rewards = batch["reward"].to(self.device)
        next_states = batch["next_state"].to(self.device)
        dones = batch["done"].to(self.device)

        # Critic update
        with torch.no_grad():
            next_action, next_logp = self.actor.sample(next_states)
            q1_next = self.critic_target_1(next_states, next_action)
            q2_next = self.critic_target_2(next_states, next_action)
            min_q_next = torch.min(q1_next, q2_next)

            target = rewards + self.gamma * (1 - dones) * (min_q_next - current_alpha * next_logp)

        q1 = self.critic_1(states, actions)
        q2 = self.critic_2(states, actions)

        critic_loss_1 = F.mse_loss(q1, target)
        critic_loss_2 = F.mse_loss(q2, target)

        self.critic_opt_1.zero_grad()
        critic_loss_1.backward()
        self.critic_opt_1.step()

        self.critic_opt_2.zero_grad()
        critic_loss_2.backward()
        self.critic_opt_2.step()
        # Actor update

        new_actions, logp = self.actor.sample(states)

        q1_update = self.critic_1(states, new_actions)
        q2_update = self.critic_2(states, new_actions)
        min_q_update = torch.min(q1_update, q2_update)

        actor_loss = (current_alpha * logp - min_q_update).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # Train Temp

        alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        return actor_loss, critic_loss_1, critic_loss_2, alpha_loss
