import copy

import numpy as np
import torch
from torch import optim, nn

from backend.CommonModels.src.Actor_ACER import Actor
from backend.CommonModels.src.Critic import Critic
from backend.Utils.src.utils import synchronize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ACERTrainer:
    def __init__(self, state_size, action_size, hidden_size, learning_rate=1e-4, gamma=0.99, tau=0.005, trust_region_delta=0.01):
        self.actor = Actor(state_size, action_size, hidden_size).to(device)

        # trust region reference policy
        self.trust_region_actor = copy.deepcopy(self.actor).to(device)
        self.critic = Critic(state_size, action_size, hidden_size).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)

        self.state_dim = state_size
        self.action_dim = action_size
        self.gamma = gamma
        self.tau = tau
        self.delta = trust_region_delta
        self.truncation_constant = 1.0
        self.seq_len = 20
        self.beta =  0.1
        self.retrace_lambda = 1.0

    def _prepare_batch(self, replay_buffer, batch_size, on_policy):
        if on_policy:
            seq = list(replay_buffer.buffer)[-self.seq_len:]
            batch = [seq]
        else:
            batch = replay_buffer.sample_sequence(self.seq_len, batch_size)

        states, actions, rewards, not_dones, mu_logps, next_states, mu_means, mu_log_stds = [], [], [], [], [], [], [], []

        for sequence in batch:
            s, a, ns, r, nd, ml, mmean, mlogstd = zip(*sequence)
            states.append(s)
            actions.append(a)
            next_states.append(ns)
            rewards.append(r)
            not_dones.append(nd)
            mu_logps.append(ml)
            mu_means.append(mmean)
            mu_log_stds.append(mlogstd)

        return (
            torch.FloatTensor(np.array(states)).to(device),
            torch.FloatTensor(np.array(actions)).to(device),
            torch.FloatTensor(np.array(rewards)).to(device),
            torch.FloatTensor(np.array(not_dones)).to(device),
            torch.FloatTensor(np.array(mu_logps)).to(device),
            torch.FloatTensor(np.array(next_states)).to(device),
            torch.FloatTensor(np.array(mu_means)).to(device),
            torch.FloatTensor(np.array(mu_log_stds)).to(device))

    def select_action(self, state, return_params=False):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        action, mu_logp, mu_mean, mu_log_std = self.actor.sample_action_with_params(state_t)
        action_np = action.detach().cpu().numpy()[0]
        mu_logp_np = mu_logp.detach().cpu().numpy()[0]
        mu_mean_np = mu_mean.detach().cpu().numpy()[0]
        mu_log_std_np = mu_log_std.detach().cpu().numpy()
        if return_params:
            return action_np, mu_logp_np, mu_mean_np, mu_log_std_np
        else:
            return action_np, mu_logp_np

    def train(self, replay_buffer, batch_size=256, on_policy=False):
        states, actions, rewards, not_dones, mu_logps, next_states, mu_means, mu_log_stds = self._prepare_batch(replay_buffer, batch_size, on_policy)

        B, T = states.shape[0], states.shape[1]

        # Critic values
        q_vals = self.critic(states.view(-1, self.state_dim),
                             actions.view(-1, self.action_dim)).view(B, T)
        with torch.no_grad():
            pi_actions, _ = self.actor.sample_action(states.view(-1, self.state_dim))
            v_vals = self.critic(states.view(-1, self.state_dim), pi_actions).view(B, T)

        # Importance weights
        policy_logp = self.actor.log_prob(states.view(-1, self.state_dim),
                                          actions.view(-1, self.action_dim)).view(B, T)
        if on_policy:
            rho = torch.ones_like(policy_logp)
            rho_bar = torch.ones_like(policy_logp)
        else:
            rho = torch.exp(policy_logp - mu_logps)
            rho_bar = torch.clamp(rho, max=self.truncation_constant)

        # Bootstrap V(s_T)
        with torch.no_grad():
            pi_last, _ = self.actor.sample_action(next_states[:, -1, :])
            v_last = self.critic(next_states[:, -1, :], pi_last).squeeze(-1)
            v_last = v_last * not_dones[:, -1]

        v_tp1 = torch.empty(B, T, device=device)
        v_tp1[:, :-1] = v_vals[:, 1:]
        v_tp1[:, -1] = v_last

        # Retrace recursion
        G = v_last.squeeze(-1).clone()
        retrace_targets = []
        for t in reversed(range(T)):
            delta_t = rewards[:, t] + not_dones[:, t] * self.gamma * v_tp1[:, t] - q_vals[:, t]
            G = q_vals[:, t] + rho_bar[:, t] * delta_t \
                + not_dones[:, t] * self.gamma * self.retrace_lambda * rho_bar[:, t] * G
            retrace_targets.insert(0, G)
        target_q = torch.stack(retrace_targets, dim=1)

        # Critic update
        critic_loss = ((q_vals - target_q.detach()) ** 2).mean()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor update
        actor_loss = 0.0
        for t in range(T):
            s_t = states[:, t, :]
            a_t = actions[:, t, :]
            q_t = self.critic(s_t, a_t).squeeze(-1)

            with torch.no_grad():
                pi_action_t, _ = self.actor.sample_action(s_t)
                v_t = self.critic(s_t, pi_action_t).squeeze(-1)

            advantage_t = (q_t - v_t).detach()
            rho_t = rho[:, t]
            rho_bar_t = torch.clamp(rho_t, max=self.truncation_constant)

            logp_t = self.actor.log_prob(s_t, a_t).squeeze(-1)
            pg_loss_t = -(rho_bar_t * logp_t * advantage_t).mean()

            # Bias correction term
            with torch.no_grad():

                a_bc, logp_bc = self.actor.sample_action(s_t)
                mu_mean_t = mu_means[:, t, :]
                mu_log_std_t = mu_log_stds[:, t, :]
                mu_std_t = mu_log_std_t.exp()

                a_bc_clamped = a_bc.clamp(-0.999, 0.999)
                raw_a_bc = 0.5 * torch.log((1 + a_bc_clamped) / (1 - a_bc_clamped + 1e-6))

                mu_dist = torch.distributions.Normal(mu_mean_t, mu_std_t)
                gaussian_log_mu_bc = mu_dist.log_prob(raw_a_bc).sum(dim=-1)
                log_det_jacobian_mu = torch.log(1 - a_bc_clamped.pow(2) + 1e-6).sum(dim=-1)
                log_mu_bc = gaussian_log_mu_bc - log_det_jacobian_mu

                rho_bc = torch.exp(logp_bc - log_mu_bc)
                q_pi_bc_t = self.critic(s_t, a_bc).squeeze(-1)
                bias_adv_t = (q_pi_bc_t - v_t).detach()
            bias_correction_t = -((rho_bc - self.truncation_constant).clamp(min=0)* bias_adv_t).mean()

            actor_loss = actor_loss + pg_loss_t + bias_correction_t

        # Trust region KL penalty
        policy_mean, policy_std = self.actor.forward(states.view(-1, self.state_dim))
        ref_mean, ref_std = self.trust_region_actor.forward(states.view(-1, self.state_dim))

        kl_t = torch.log(ref_std / policy_std) + (policy_std ** 2 + (policy_mean - ref_mean) ** 2) / (
                    2.0 * ref_std ** 2) - 0.5
        kl_t = kl_t.sum(dim=-1).view(B, T)


        # Entropy regularization
        dist = torch.distributions.Normal(policy_mean, policy_std)
        entropy = dist.entropy().sum(dim=-1).mean()
        actor_loss += -self.beta * entropy

        # apply actor update
        self.actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)

        kl = (kl_t * not_dones).mean()
        k_grad = torch.autograd.grad(kl, self.actor.parameters(), retain_graph=True)

        # PROJECTION STEP
        g = [p.grad for p in self.actor.parameters()]
        k_dot_g = sum((kg * gg).sum() for kg, gg in zip(k_grad, g))
        k_norm_sq = sum((kg * kg).sum() for kg in k_grad) + 1e-8

        alpha = torch.clamp((k_dot_g - self.delta) / k_norm_sq, min=0.0)

        for p, kg in zip(self.actor.parameters(), k_grad):
            if p.grad is None:
                continue
            p.grad -= alpha * kg

        self.actor_optimizer.step()

        # update average policy (trust region reference)
        synchronize(self.actor, self.trust_region_actor, tau=self.tau)