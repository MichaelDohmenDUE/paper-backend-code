import torch
import torch.nn.functional as F
from torch import nn

from backend.Utils.src.NodeLib.NodeLibrary import bellman, optimizer_update, detransition, mean_squared_error, \
    deterministic_policy_gradient
from backend.Utils.src.ReplayBuffer import ReplayBuffer


class TrainProcess:
    def __init__(self, replay_buffer: ReplayBuffer, actor: nn.Module, actor_target: nn.Module, critic: nn.Module,
                 critic_target: nn.Module, actor_optimizer: torch.optim.Optimizer,
                 critic_optimizer: torch.optim.Optimizer, gamma: float, warmup: int, device: torch.device):
        self.replay_buffer = replay_buffer
        self.actor = actor.to(device)
        self.actor_target = actor_target.to(device)
        self.critic = critic.to(device)
        self.critic_target = critic_target.to(device)
        self.actor_opt = actor_optimizer
        self.critic_opt = critic_optimizer
        self.gamma = gamma
        self.device = device

    def run(self):
        if len(self.replay_buffer) < self.replay_buffer.batch_size:
            return {}
        batch = self.replay_buffer.sample_batch()
        states, actions, rewards, next_states, dones = detransition(self.replay_buffer.spec.fields, batch, self.device)
        # Critic update
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            target_q = target_q.squeeze()
            target = bellman(target_Q=target_q, reward=rewards, done=dones, discount_factor=self.gamma)
        current_q = self.critic(states, actions).squeeze()
        critic_loss = mean_squared_error(current_q, target)
        optimizer_update(optimizer=self.critic_opt, loss=critic_loss)

        # Actor update
        action = self.actor(states)
        q_value = self.critic(states, action)
        actor_loss = deterministic_policy_gradient(q_value)
        optimizer_update(optimizer=self.actor_opt, loss=actor_loss)
        metrics = {
            "losses/critic_loss": critic_loss.item(),
            "losses/actor_loss": actor_loss.item(),
        }
        return metrics
