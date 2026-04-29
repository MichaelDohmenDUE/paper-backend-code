import torch
import torch.nn.functional as F
from torch import nn

from backend.StochasticPolicy.A3C.src.A3CNodes import optimizer_step
from backend.Utils.src.NodeLib.NodeLibrary import detransition, optimizer_normalized, optimizer_update, soft_bellman
from backend.Utils.src.ReplayBuffer import ReplayBuffer


class TrainProcessor:
    def __init__(self, buffer: ReplayBuffer, behaviour: nn.Module, critic_1: nn.Module,
                 critic_target_1: nn.Module, critic_2: nn.Module, critic_target_2: nn.Module,
                 actor_optimizer: torch.optim.Optimizer,
                 critic_optimizer_1: torch.optim.Optimizer, critic_optimizer_2: torch.optim.Optimizer,
                 log_alpha, alpha_optimizer, target_entropy, gamma, device):
        self.target_entropy = target_entropy
        self.log_alpha = log_alpha
        self.alpha_optimizer = alpha_optimizer
        self.buffer = buffer
        self.behaviour = behaviour.to(device)
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
        if len(self.buffer) < 10000:#self.buffer.batch_size:
            return {}

        current_alpha = self.log_alpha.exp()

        batch = self.buffer.sample_batch()
        states, actions, rewards, next_states, dones = detransition(self.buffer.spec.fields, batch, self.device)

        # Critic update
        with torch.no_grad():
            next_action, next_logp = self.behaviour.sample(next_states)
            next_logp = next_logp.reshape(-1)
            q1_next = self.critic_target_1(next_states, next_action).reshape(-1)
            q2_next = self.critic_target_2(next_states, next_action).reshape(-1)
            min_q_next = torch.min(q1_next, q2_next)
            target = soft_bellman(min_q_next, rewards, dones, self.gamma, current_alpha, next_logp)

        q1 = self.critic_1(states, actions).reshape(-1)
        q2 = self.critic_2(states, actions).reshape(-1)

        critic_loss_1 = F.mse_loss(q1, target)
        critic_loss_2 = F.mse_loss(q2, target)

        optimizer_update(self.critic_opt_1, critic_loss_1)

        optimizer_update(self.critic_opt_2, critic_loss_2)
        # behaviour update

        new_actions, logp = self.behaviour.sample(states)
        logp = logp.reshape(-1)
        q1_update = self.critic_1(states, new_actions).reshape(-1)
        q2_update = self.critic_2(states, new_actions).reshape(-1)
        min_q_update = torch.min(q1_update, q2_update)

        actor_loss = (current_alpha * logp - min_q_update).mean()

        optimizer_update(self.actor_opt, actor_loss)
        # Train Temp

        alpha_loss = -(self.log_alpha.exp() * (logp + self.target_entropy).detach()).mean()
        optimizer_update(self.alpha_optimizer, alpha_loss)

        metrics = {
            "losses/actor_loss": actor_loss.item(),
            "losses/critic_1_loss": critic_loss_1.item(),
            "losses/critic_2_loss": critic_loss_2.item(),
            "losses/temp_loss": alpha_loss.item(),
            "val/q1": torch.mean(q1.detach()),
        }
        return metrics
