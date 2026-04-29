import numpy as np
import torch
from torch import nn

from backend.Utils.src.RolloutBuffer import RolloutBuffer
from backend.Utils.src.NodeLib.NodeLibrary import detransition, td_residual, normalize, clipped_surrogate_objective
from backend.Utils.src.utils import gae

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PPOTrainerProcessor:
    def __init__(self, actor, optimizer, rollout_buffer: RolloutBuffer, batch_size: int = 64, epochs: int = 10,
                 clip_eps=0.2, vf_coef=1.0, ent_coef=0.01, max_grad_norm=0.5, gamma=0.99, lam=0.95):
        self.actor = actor
        self.optimizer = optimizer
        self.rollout_buffer = rollout_buffer
        self.batch_size = batch_size
        self.epochs = epochs
        self.clip_eps = clip_eps
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.use_value_clip = True  # TODO: Spinup Implementation uses value Clipping, original PPO Paper does not
        self.gamma = gamma
        self.lam = lam
        self.device = device

    def run(self):
        # Logging
        all_policy_losses = []
        all_value_losses = []
        all_entropies = []
        all_approx_kls = []

        #######
        if not self.rollout_buffer.reached_rollout_size():
            return {}

        rollout = self.rollout_buffer.sample()
        states, actions, logps, rewards, dones, values, bootstrap_values = detransition(
            self.rollout_buffer.spec.fields, rollout, self.device
        )
        #print("#######", states.shape, actions.shape, logps.shape, rewards.shape, values.shape, bootstrap_values.shape)
        r = rewards.view(-1, self.rollout_buffer.num_envs)
        d = dones.view(-1, self.rollout_buffer.num_envs)
        v = values.view(-1, self.rollout_buffer.num_envs)
        b_vals = bootstrap_values.view(-1, self.rollout_buffer.num_envs)
        all_advantages = torch.zeros_like(v)
        for i in range(self.rollout_buffer.num_envs):
            boot_value = b_vals[-1, i].unsqueeze(0)
            deltas = td_residual(r[:, i], d[:, i], v[:, i], boot_value, self.gamma)
            all_advantages[:, i] = gae(self.gamma, self.lam, deltas, d[:, i])

        advantages = all_advantages.reshape(-1)
        returns = advantages + values
        advantages = normalize(advantages)

        dataset_size = len(states)
        indices = np.arange(dataset_size)

        for epoch in range(self.epochs):
            np.random.shuffle(indices)

            for start in range(0, dataset_size, self.batch_size):
                batch_idx = indices[start:start + self.batch_size]
                b_states = states[batch_idx]
                b_actions = actions[batch_idx]
                b_old_logps = logps[batch_idx]
                b_adv = advantages[batch_idx]
                b_ret = returns[batch_idx]

                dist, value = self.actor(b_states)
                new_logp = dist.log_prob(b_actions)
                entropy = dist.entropy().mean()
                value_pred = value.squeeze(-1)

                # Policy Losses
                # print(new_logp.shape, b_old_logps.shape, b_adv.shape, value_pred.shape)
                policy_loss = clipped_surrogate_objective(new_logp, b_old_logps, b_adv, self.clip_eps)

                # value loss
                # print(b_ret.shape, value_pred.shape, entropy.shape)
                value_loss = 0.5 * (b_ret - value_pred).pow(2).mean()

                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy

                # Backprop
                self.optimizer.zero_grad()
                loss.backward()

                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # with torch.no_grad():
                #    approx_kl = ((ratio - 1) - torch.log(ratio)).mean().item()

                all_policy_losses.append(policy_loss.item())
                all_value_losses.append(value_loss.item())
                all_entropies.append(entropy.item())
                # all_approx_kls.append(approx_kl)

        return {
            "losses/value_loss": np.mean(all_value_losses),
            "losses/policy_loss": np.mean(all_policy_losses),
            "losses/entropy": np.mean(all_entropies),
        }
