import copy

import numpy as np
import torch

from TobeTranslatedAlgorithms.ACER.src.continuos.Actor_ACER import Actor
from DeterministicPolicy.DDPG.src.Critic import Critic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ACERTrainer:
    def __init__(self, state_size, action_size, hidden_size, learning_rate=1e-4, gamma=0.99, tau=0.005,
                 trust_region_delta=0.01):
        self.actor = Actor(state_size, action_size, hidden_size).to(device)
        self.trust_region_actor = copy.deepcopy(self.actor).to(device)
        self.critic = Critic(state_size, action_size, hidden_size).to(device)
        self.actor_optimizer = torch.optim.RMSprop(self.actor.parameters(), lr=7e-4, alpha=0.99, eps=1e-5)
        self.critic_optimizer = torch.optim.RMSprop(self.critic.parameters(), lr=7e-4, alpha=0.99, eps=1e-5)


        self.state_dim = state_size
        self.action_dim = action_size
        self.gamma = gamma
        self.tau = 0.01
        self.delta = trust_region_delta
        self.rho_bar = 5.0
        self.c_bar = 1.0
        self.seq_len = 20
        self.retrace_lambda = 1.0
        self.entropy_coeff = 0.001

    def _prepare_batch(self, replay_buffer, batch_size, on_policy):
        if on_policy:
            seq = list(replay_buffer.buffer)[-self.seq_len:]

            states = torch.tensor(np.array([[tr.state for tr in seq]]), dtype=torch.float32, device=device)
            actions = torch.tensor(np.array([[tr.action for tr in seq]]), dtype=torch.float32, device=device)
            rewards = torch.tensor(np.array([[tr.reward for tr in seq]]), dtype=torch.float32, device=device)
            not_dones = torch.tensor(np.array([[tr.mask for tr in seq]]), dtype=torch.float32, device=device)
            next_states = torch.tensor(np.array([[tr.next_state for tr in seq]]), dtype=torch.float32, device=device)
            mu_logps = torch.tensor(np.array([[tr.mu_logp for tr in seq]]), dtype=torch.float32, device=device)
            mu_means = torch.tensor(np.array([[tr.mu_mean for tr in seq]]), dtype=torch.float32, device=device)
            mu_log_stds = torch.tensor(np.array([[tr.mu_log_std for tr in seq]]), dtype=torch.float32, device=device)

            return states, actions, rewards, not_dones, mu_logps, next_states, mu_means, mu_log_stds

        else:
            batch = replay_buffer.sample_sequence_batch(self.seq_len, batch_size)

            states = batch["state"].to(device)
            actions = batch["action"].to(device)
            rewards = batch["reward"].to(device).squeeze(-1)
            not_dones = batch["mask"].to(device).squeeze(-1)
            next_states = batch["next_state"].to(device)
            mu_logps = batch["mu_logp"].to(device).squeeze(-1)
            mu_means = batch["mu_mean"].to(device)
            mu_log_stds = batch["mu_log_std"].to(device)

            return states, actions, rewards, not_dones, mu_logps, next_states, mu_means, mu_log_stds

    def _compute_q_values(self, states, actions):
        return self.critic(states, actions)

    def _compute_v_values(self, states):
        pi_actions, _ = self.actor.sample_action(states)
        return self.critic(states, pi_actions)

    def _compute_importance_weights(self, policy_logp, mu_logps):
        log_ratio = (policy_logp - mu_logps).clamp(-10, 10)
        rho = torch.exp(log_ratio).clamp(min=1e-7)
        rho_bar = rho.clamp(max=self.rho_bar)
        c = rho.clamp(max=self.c_bar)
        return rho, rho_bar, c

    def _compute_bootstrap_value(self, next_states, not_dones):
        with torch.no_grad():
            pi_last, _ = self.actor.sample_action(next_states)
            v_last = self.critic(next_states, pi_last).squeeze(-1)
            return v_last * not_dones

    def _compute_retrace_targets(self, rewards, not_dones, q_vals, v_tp1, c):
        B, T = rewards.shape
        G = v_tp1[:, -1].clone()
        retrace_targets = []
        for t in reversed(range(T)):
            q_t = q_vals[:, t]
            v_tp1_t = v_tp1[:, t]
            r_t = rewards[:, t]
            nd_t = not_dones[:, t]
            c_t = c[:, t]
            delta_t = r_t + nd_t * self.gamma * v_tp1_t - q_t
            delta_t = delta_t.clamp(-10.0, 10.0)
            G = q_t + c_t * (delta_t + nd_t * self.gamma * self.retrace_lambda * (G - v_tp1_t))
            G = G.clamp(-100.0, 100.0)
            retrace_targets.insert(0, G)
        return torch.stack(retrace_targets, dim=1)

    def _compute_q_opc(self, rewards, not_dones, v_vals):
        B, T = rewards.shape
        Q_opc = torch.zeros_like(rewards, device=device)
        G = v_vals[:, -1].clone()
        for t in reversed(range(T)):
            r_t = rewards[:, t]
            nd_t = not_dones[:, t]
            G = r_t + nd_t * self.gamma * G
            Q_opc[:, t] = G
        return Q_opc

    def _critic_loss_function(self, q_vals, target_q):
        return ((q_vals - target_q.detach()) ** 2).mean()

    def _update_critic(self, loss):
        self.critic_optimizer.zero_grad()
        loss.backward()
        # NOT IN ACER PSEUDOCODE — numerical stability
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
        self.critic_optimizer.step()

    def _policy_gradient(self, s_t, a_t, rho_bar_t, advantage_t):
        logp_t = self.actor.log_prob(s_t, a_t).squeeze(-1)
        return -(rho_bar_t * logp_t * advantage_t).mean()

    def _bias_correction(self, s_t, mu_mean_t, mu_log_std_t, v_t):
        with torch.no_grad():
            a_bc, logp_bc = self.actor.sample_action(s_t)
            mu_std_t = mu_log_std_t.exp()
            mu_dist = torch.distributions.Normal(mu_mean_t, mu_std_t)
            logp_mu = mu_dist.log_prob(a_bc).sum(dim=-1)
            log_ratio_bc = (logp_bc - logp_mu).clamp(-20, 20)
            rho_bc = torch.exp(log_ratio_bc).clamp(0.0, self.rho_bar)
            q_pi_bc_t = self.critic(s_t, a_bc).squeeze(-1)
            bias_adv_t = (q_pi_bc_t - v_t).detach()
        return -((rho_bc - self.rho_bar).clamp(min=0) * bias_adv_t).mean()

    def _compute_kl(self, policy_mean, policy_std, ref_mean, ref_std):
        policy_std = policy_std.clamp(1e-3, 10.0)
        ref_std = ref_std.clamp(1e-3, 10.0)
        mean_diff = (policy_mean - ref_mean).clamp(-10.0, 10.0)
        kl = torch.log(ref_std / policy_std) + (policy_std ** 2 + mean_diff ** 2) / (2 * ref_std ** 2) - 0.5
        return kl.sum(dim=-1)

    def _trust_region_projection(self, kl_grad):
        kl_grad = [torch.clamp(g, -1.0, 1.0) for g in kl_grad]
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
        # NOT IN ACER PSEUDOCODE — gradient clipping
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=10.0)
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
        states, actions, rewards, not_dones, mu_logps, next_states, mu_means, mu_log_stds = self._prepare_batch(
            replay_buffer, batch_size, on_policy)

        B, T = states.shape[0], states.shape[1]

        flat_states = states.view(-1, self.state_dim)
        flat_actions = actions.view(-1, self.action_dim)

        q_vals = self._compute_q_values(flat_states, flat_actions).view(B, T)

        with torch.no_grad():
            v_vals = self._compute_v_values(flat_states).view(B, T)

        policy_logp = self.actor.log_prob(flat_states, flat_actions).view(B, T)
        rho, rho_bar, c = self._compute_importance_weights(policy_logp, mu_logps)

        v_last = self._compute_bootstrap_value(next_states[:, -1, :], not_dones[:, -1])
        v_tp1 = torch.empty(B, T, device=device)
        v_tp1[:, :-1] = v_vals[:, 1:]
        v_tp1[:, -1] = v_last

        target_q = self._compute_retrace_targets(rewards, not_dones, q_vals, v_tp1, c)

        critic_loss = self._critic_loss_function(q_vals, target_q) / T
        self._update_critic(critic_loss)

        Q_opc = self._compute_q_opc(rewards, not_dones, v_vals)
        actor_loss = 0.0

        _, _, policy_mean, policy_log_std = self.actor.sample_action_with_params(flat_states)
        policy_std = policy_log_std.exp()
        dist = torch.distributions.Normal(policy_mean, policy_std)
        entropy = dist.entropy().sum(dim=-1).mean()

        for t in range(T):
            s_t = states[:, t, :]
            a_t = actions[:, t, :]
            rho_bar_t = rho_bar[:, t]

            q_opc_t = Q_opc[:, t]
            v_t = v_vals[:, t]
            advantage_t = (q_opc_t - v_t).detach()
            advantage_t = torch.clamp(advantage_t, -10.0, 10.0)

            pg_loss_t = self._policy_gradient(s_t, a_t, rho_bar_t, advantage_t)
            bc_loss_t = self._bias_correction(s_t, mu_means[:, t, :], mu_log_stds[:, t, :], v_t)

            actor_loss = actor_loss + pg_loss_t + bc_loss_t

        actor_loss = (actor_loss / T) - (self.entropy_coeff * entropy)

        _, _, ref_mean, ref_log_std = self.trust_region_actor.sample_action_with_params(flat_states)
        ref_std = ref_log_std.exp()
        kl = self._compute_kl(policy_mean, policy_std, ref_mean, ref_std).view(B, T)
        kl = (kl * not_dones).mean()
        kl_grad = torch.autograd.grad(kl, self.actor.parameters(), retain_graph=True)

        self._update_actor(actor_loss, kl_grad=kl_grad)
        with torch.no_grad():
            for p, ap in zip(self.actor.parameters(), self.trust_region_actor.parameters()):
                ap.data.copy_(ap.data * (1.0 - self.tau) + p.data * self.tau)

        if torch.rand(1).item() < 0.001:
            print(
                f"critic_loss={critic_loss.item():.3f}, "
                f"actor_loss={actor_loss.item():.3f}, "
                f"entropy={entropy.item():.3f}, "  # --- CHANGE: Monitoring entropy ---
                f"mean_rho={rho.mean().item():.3f}, "
                f"mean_c={c.mean().item():.3f},"
                f"mean_kl={kl.item():.5f}")