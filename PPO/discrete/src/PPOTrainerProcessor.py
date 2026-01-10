import numpy as np
import torch
from torch import nn

from backend.Utils.src.utils import compute_gae

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PPOTrainerProcessor:
    def __init__(self, actor, critic, optimizer, replay_buffer, batch_size: int = 64, epochs: int = 10,
                 clip_eps=0.2, vf_coef=1.0, ent_coef=0.01, max_grad_norm=0.5, gamma=0.99, lam=0.95):
        self.actor = actor
        self.critic = critic
        self.optimizer = optimizer
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size
        self.epochs = epochs
        self.clip_eps = clip_eps
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.use_value_clip = False  # TODO: Spinup Implementation uses value Clipping, original PPO Paper does not
        self.gamma = gamma
        self.lam = lam
        self.device = device

    def run(self):
        states, actions, old_logps, advantages, returns = compute_gae(self.replay_buffer, gamma=self.gamma, lam=self.lam)

        # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Convert to tensors
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        old_logps = torch.as_tensor(old_logps, dtype=torch.float32, device=self.device)
        advantages = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(returns, dtype=torch.float32, device=self.device)

        # Replaybuffer Rollout
        replaybuffer_rollout = list(zip(states, actions, old_logps, advantages, returns))

        for _ in range(self.epochs):
            np.random.shuffle(replaybuffer_rollout)
            for start in range(0, len(replaybuffer_rollout), self.batch_size):
                batch = replaybuffer_rollout[start:start + self.batch_size]
                b_states, b_actions, b_old_logps, b_adv, b_ret = zip(*batch)
                b_states = torch.stack(b_states)
                b_actions = torch.stack(b_actions).long()
                b_old_logps = torch.stack(b_old_logps)
                b_adv = torch.stack(b_adv)
                b_ret = torch.stack(b_ret)

                dist = self.actor(b_states)
                new_logp = dist.log_prob(b_actions)
                entropy = dist.entropy().mean()
                value_pred = self.critic(b_states).squeeze(-1)

                # Policy Losses
                ratio = torch.exp(new_logp - b_old_logps)
                surrogate_objective = ratio * b_adv
                surrogate_objective2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * b_adv
                policy_loss = -torch.min(surrogate_objective, surrogate_objective2).mean()

                # value loss
                value_loss = (b_ret - value_pred).pow(2).mean()

                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy
                #print(f"total loss {loss}")

                # Backprop
                self.optimizer.zero_grad()
                loss.backward()
                if self.use_value_clip: # NOT USED BY ORIGINAL PPO
                    nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()),self.max_grad_norm)
                self.optimizer.step()

