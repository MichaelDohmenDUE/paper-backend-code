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
        all_policy_losses = []
        all_value_losses = []
        all_entropies = []
        all_approx_kls = []
        states, actions, old_logps, advantages, returns = compute_gae(self.replay_buffer, gamma=self.gamma,
                                                                      lam=self.lam)

       # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Convert to tensors
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        old_logps = torch.as_tensor(old_logps, dtype=torch.float32, device=self.device)
        advantages = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(returns, dtype=torch.float32, device=self.device)

        dataset_size = len(states)
        indices = np.arange(dataset_size)

        for epoch in range(self.epochs):
            # 3. Shuffle indices, not data
            np.random.shuffle(indices)

            for start in range(0, dataset_size, self.batch_size):
                batch_idx = indices[start:start + self.batch_size]
                b_states = states[batch_idx]
                b_actions = actions[batch_idx]
                b_old_logps = old_logps[batch_idx]
                b_adv = advantages[batch_idx]
                b_ret = returns[batch_idx]

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
                value_loss =  0.5 * (b_ret - value_pred).pow(2).mean()

                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy

                # Backprop
                self.optimizer.zero_grad()
                loss.backward()
                if self.use_value_clip:  # NOT USED BY ORIGINAL PPO
                    nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()),
                                             self.max_grad_norm)
                self.optimizer.step()
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - torch.log(ratio)).mean().item()

                all_policy_losses.append(policy_loss.item())
                all_value_losses.append(value_loss.item())
                all_entropies.append(entropy.item())
                all_approx_kls.append(approx_kl)

        return {
            "losses/value_loss": np.mean(all_value_losses),
            "losses/policy_loss": np.mean(all_policy_losses),
            "losses/entropy": np.mean(all_entropies),
            "losses/approx_kl": np.mean(all_approx_kls)
        }