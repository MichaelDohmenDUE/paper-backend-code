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

    def _compute_q_values(self, states, actions):
        return self.critic(states, actions)

    def _compute_v_values(self, states):
        pi_actions, _ = self.actor.sample_action(states)
        return self.critic(states, pi_actions)

    def _compute_importance_weights(self, policy_logp, mu_logps, on_policy):
        if on_policy:
            rho = torch.ones_like(policy_logp)
            rho_bar = torch.ones_like(policy_logp)
        else:
            rho = torch.exp(policy_logp - mu_logps)
            rho_bar = torch.clamp(rho, max=self.truncation_constant)
        return rho, rho_bar

    def _compute_bootstrap_value(self, next_states, not_dones):
        with torch.no_grad():
            pi_last, _ = self.actor.sample_action(next_states)
            v_last = self.critic(next_states, pi_last).squeeze(-1)
            return v_last * not_dones

    def _compute_retrace_targets(self, rewards, not_dones, q_vals, v_tp1, rho_bar):
        B, T = rewards.shape
        G = v_tp1[:, -1].clone()
        retrace_targets = []
        for t in reversed(range(T)):
            delta_t = rewards[:, t] + not_dones[:, t] * self.gamma * v_tp1[:, t] - q_vals[:, t]
            G = q_vals[:, t] + rho_bar[:, t] * delta_t  + not_dones[:, t] * self.gamma * self.retrace_lambda * rho_bar[:, t] * G
            retrace_targets.insert(0, G)
        return torch.stack(retrace_targets, dim=1)

    def _critic_loss_function(self, q_vals, target_q):
        return ((q_vals - target_q.detach()) ** 2).mean()

    def _update_critic(self, loss):
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

    def _policy_gradient(self, s_t, a_t, rho_bar_t, advantage_t):
        logp_t = self.actor.log_prob(s_t, a_t).squeeze(-1)
        return -(rho_bar_t * logp_t * advantage_t).mean()

    def _bias_correction(self, s_t, mu_mean_t, mu_log_std_t, v_t):
        with torch.no_grad():
            a_bc, logp_bc = self.actor.sample_action(s_t)
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
        return -((rho_bc - self.truncation_constant).clamp(min=0) * bias_adv_t).mean()

    def _entropy_correction(self, policy_mean, policy_std):
        dist = torch.distributions.Normal(policy_mean, policy_std)
        return dist.entropy().sum(dim=-1).mean()

    def _compute_kl(self, policy_mean, policy_std, ref_mean, ref_std):
        kl = torch.log(ref_std / policy_std) + (policy_std ** 2 + (policy_mean - ref_mean) ** 2) / (2 * ref_std ** 2) - 0.5
        return kl.sum(dim=-1)

    def _trust_region_projection(self, kl_grad):
        g = [p.grad for p in self.actor.parameters()]
        k_dot_g = sum((kg * gg).sum() for kg, gg in zip(kl_grad, g))
        k_norm_sq = sum((kg * kg).sum() for kg in kl_grad) + 1e-8
        alpha = torch.clamp((k_dot_g - self.delta) / k_norm_sq, min=0.0)

        for p, kg in zip(self.actor.parameters(), kl_grad):
            if p.grad is not None:
                p.grad -= alpha * kg

    def _update_actor(self, actor_loss, kl_grad):
        self.actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self._trust_region_projection(kl_grad)
        self.actor_optimizer.step()

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

        flat_states = states.view(-1, self.state_dim)
        flat_actions = actions.view(-1, self.action_dim)

        # Critic values
        q_vals = self._compute_q_values(flat_states, flat_actions).view(B, T)

        with torch.no_grad():
            v_vals = self._compute_v_values(flat_states).view(B, T)

        # Importance weights

        policy_logp = self.actor.log_prob(flat_states, flat_actions).view(B, T)
        rho, rho_bar = self._compute_importance_weights(policy_logp, mu_logps, on_policy)

        # Bootstrap V(s_T)
        v_last = self._compute_bootstrap_value(next_states[:, -1, :], not_dones[:, -1])
        v_tp1 = torch.empty(B, T, device=device)
        v_tp1[:, :-1] = v_vals[:, 1:]
        v_tp1[:, -1] = v_last

        # Retrace recursion
        target_q = self._compute_retrace_targets(rewards, not_dones, q_vals, v_tp1, rho_bar)

        # Critic update
        critic_loss = self._critic_loss_function(q_vals, target_q)
        self._update_critic(critic_loss)


        # Actor update
        actor_loss = torch.zeros((), device=device)
        for t in range(T):
            s_t = states[:, t, :]
            a_t = actions[:, t, :]
            rho_bar_t = rho_bar[:, t]

            q_t = q_vals[:, t]
            v_t = v_vals[:, t]
            advantage_t = (q_t - v_t).detach()

            pg_loss_t = self._policy_gradient(s_t, a_t, rho_bar_t, advantage_t)

            bias_correction_t = self._bias_correction( s_t, mu_means[:, t, :], mu_log_stds[:, t, :], v_t )

            actor_loss = actor_loss + pg_loss_t + bias_correction_t

        # Entropy regularization
        policy_mean, policy_std = self.actor(flat_states)
        entropy = self._entropy_correction(policy_mean, policy_std)
        actor_loss += -self.beta * entropy

        # Trust region KL
        ref_mean, ref_std = self.trust_region_actor(flat_states)
        kl = self._compute_kl(policy_mean, policy_std, ref_mean, ref_std).view(B, T)
        kl = (kl * not_dones).mean()
        kl_grad = torch.autograd.grad(kl, self.actor.parameters(), retain_graph=True)

        self._update_actor(actor_loss, kl_grad)

        synchronize(self.actor, self.trust_region_actor, tau=self.tau)