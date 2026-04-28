import numpy as np
import torch
from torch import nn

from backend.Utils.src.NodeLib.NodeLibrary import detransition, td_residual, normalize, clipped_surrogate_objective
from backend.Utils.src.utils import compute_gae, gae

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PPOTrainerProcessor:
    def __init__(self, actor, critic, optimizer, rollout_buffer, batch_size: int = 64, epochs: int = 10,
                 clip_eps=0.2, vf_coef=1.0, ent_coef=0.01, max_grad_norm=0.5, gamma=0.99, lam=0.95):
        self.actor = actor
        self.critic = critic
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
        #Logging
        all_policy_losses = []
        all_value_losses = []
        all_entropies = []
        all_approx_kls = []

        #######
        rollout = self.rollout_buffer.sample()
        states, actions, logps, rewards, dones, value, bootstrap_value = detransition(self.rollout_buffer.spec.fields,
                                                                                      rollout, self.device)
        #print("#######",states.shape, actions.shape, logps.shape, rewards.shape, value.shape, bootstrap_value.shape)
        deltas = td_residual(rewards, dones, value, bootstrap_value, self.gamma)

        advantages = gae(self.gamma, self.lam, deltas, dones)
        returns = advantages + value

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

                dist = self.actor(b_states)
                new_logp = dist.log_prob(b_actions)
                entropy = dist.entropy().mean()
                value_pred = self.critic(b_states).squeeze(-1)

                # Policy Losses
                #print(new_logp.shape, b_old_logps.shape, b_adv.shape, value_pred.shape)
                policy_loss = clipped_surrogate_objective(new_logp, b_old_logps, b_adv, self.clip_eps)

                # value loss
                #print(b_ret.shape, value_pred.shape, entropy.shape)
                value_loss = 0.5 * (b_ret - value_pred).pow(2).mean()

                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy

                # Backprop
                self.optimizer.zero_grad()
                loss.backward()

                nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()),self.max_grad_norm)
                self.optimizer.step()

                #with torch.no_grad():
                #    approx_kl = ((ratio - 1) - torch.log(ratio)).mean().item()

                all_policy_losses.append(policy_loss.item())
                all_value_losses.append(value_loss.item())
                all_entropies.append(entropy.item())
                #all_approx_kls.append(approx_kl)

        return {
            "losses/value_loss": np.mean(all_value_losses),
            "losses/policy_loss": np.mean(all_policy_losses),
            "losses/entropy": np.mean(all_entropies),
        }
