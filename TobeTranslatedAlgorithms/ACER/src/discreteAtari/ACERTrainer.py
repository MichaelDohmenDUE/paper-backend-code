import copy

import numpy as np
import torch
import torch.distributions
from TobeTranslatedAlgorithms.ACER.src.discreteAtari.ActorACerMujoco import ACERNet
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ACERTrainer:
    def __init__(self, state_size, action_size, hidden_size, learning_rate=1e-4, gamma=0.99, tau=0.005,
                 trust_region_delta=0.01):
        self.model = ACERNet(action_size).to(device)
        self.trust_region_model = copy.deepcopy(self.model).to(device)
        self.optimzer = torch.optim.RMSprop(self.model.parameters(), lr=7e-4, alpha=0.99, eps=10e-5)

        self.state_dim = state_size
        self.action_dim = action_size
        self.gamma = gamma
        self.tau = tau
        self.delta = trust_region_delta
        self.rho_bar = 10.0
        self.c_bar = 1.0
        self.seq_len = 20
        self.retrace_lambda = 1.0
        self.entropy_scale = 0.01

    def _prepare_batch(self, replay_buffer, batch_size, on_policy):
        if on_policy:
            rollouts = replay_buffer  # list of B rollouts, each length T

            # Convert nested lists to NumPy arrays first (fast)
            states_np = np.array([[tr.state for tr in rollout] for rollout in rollouts])
            actions_np = np.array([[tr.action for tr in rollout] for rollout in rollouts])
            rewards_np = np.array([[tr.reward for tr in rollout] for rollout in rollouts])
            not_dones_np = np.array([[tr.mask for tr in rollout] for rollout in rollouts])
            next_states_np = np.array([[tr.next_state for tr in rollout] for rollout in rollouts])
            mu_logps_np = np.array([[tr.mu_logp for tr in rollout] for rollout in rollouts])
            mu_logits_np = np.array([[tr.mu_logits for tr in rollout] for rollout in rollouts])
            states = torch.tensor(states_np, device=device, dtype=torch.float32)
            actions = torch.tensor(actions_np, device=device, dtype=torch.long)
            rewards = torch.tensor(rewards_np, device=device, dtype=torch.float32)
            not_dones = torch.tensor(not_dones_np, device=device, dtype=torch.float32)
            next_states = torch.tensor(next_states_np, device=device, dtype=torch.float32)
            mu_logps = torch.tensor(mu_logps_np, device=device, dtype=torch.float32)
            mu_logits = torch.tensor(mu_logits_np, device=device, dtype=torch.float32)
            # Ensure channel-first (B, T, C, H, W) DEBUGGING with Help of LLM by Uni
            if states.ndim == 5:
                if states.shape[2] == 4:
                    # Already (B, T, C, H, W) → do nothing
                    pass
                elif states.shape[-1] == 4:  # (B, T, H, W, C)
                    states = states.permute(0, 1, 4, 2, 3)
                    next_states = next_states.permute(0, 1, 4, 2, 3)
                elif states.shape[-2] == 4:  # (B, T, H, C, W)
                    states = states.permute(0, 1, 3, 2, 4)
                    next_states = next_states.permute(0, 1, 3, 2, 4)
                else:
                    raise RuntimeError(f"Cannot find channel dimension in states, shape={states.shape}")  #DEBUGGING with Help of LLM by Uni
            return states, actions, rewards, not_dones, mu_logps, next_states, mu_logits

        else:
            batch = replay_buffer.sample_sequence_batch(self.seq_len, batch_size)

            states = batch["state"].to(device).float()
            next_states = batch["next_state"].to(device).float()
            actions = batch["action"].long().to(device)
            rewards = batch["reward"].to(device).squeeze(-1)
            not_dones = batch["mask"].to(device).squeeze(-1)
            mu_logps = batch["mu_logp"].to(device).squeeze(-1)
            mu_logits = batch["mu_logits"].to(device)

            return states, actions, rewards, not_dones, mu_logps, next_states, mu_logits

    def _compute_q_values(self, states, actions):
        _, q_all = self.model(states)
        q_sa = q_all.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        return q_sa

    def _compute_bootstrap_value_stable(self, next_states, not_dones):
        with torch.no_grad():
            logits, q_all = self.trust_region_model(next_states)
            pi = F.softmax(logits, dim=-1)
            v_last = (pi * q_all).sum(dim=-1)
            return v_last * not_dones

    def _compute_v_values(self, states, use_trust_model=False):
        if use_trust_model:
            model = self.trust_region_model
        else:
            model = self.model
        with torch.set_grad_enabled(not use_trust_model):
            logits, q_all = model(states)
            logits = torch.clamp(logits, -20, 20)
            pi = F.softmax(logits, dim=-1)
            v = (pi * q_all).sum(dim=-1)
        return v

    def _compute_importance_weights(self, policy_logp, mu_logps):
        log_ratio = (policy_logp - mu_logps).clamp(-10, 10)
        rho = torch.exp(log_ratio)
        rho_bar = rho.clamp(max=self.rho_bar)
        c = rho.clamp(max=self.c_bar)
        return rho, rho_bar, c

    def _compute_bootstrap_value(self, next_states, not_dones):
        with torch.no_grad():
            logits, q_all = self.model(next_states)
            logits = torch.clamp(logits, -20, 20)
            dist = torch.distributions.Categorical(logits=logits)
            pi = dist.probs
            v_last = (pi * q_all).sum(dim=-1)
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
            #delta_t = delta_t.clamp(-10.0, 10.0)
            G = q_t + c_t * (delta_t + nd_t * self.gamma * self.retrace_lambda * (G - v_tp1_t))
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

    def _policy_gradient(self, s_t, a_t, rho_bar_t, advantage_t):
        logp_t = self.model.log_prob(s_t, a_t).squeeze(-1)
        return -(rho_bar_t * logp_t * advantage_t).mean()

    def _compute_kl(self, logits, ref_logits):
        p = torch.softmax(logits, dim=-1)
        q = torch.softmax(ref_logits, dim=-1)
        return (p * (p.log() - q.log())).sum(dim=-1)

    def select_action(self, state, return_params=False):
        state_t = torch.from_numpy(state).unsqueeze(0).float().to(device)
        logits, _ = self.model(state_t)
        logits = torch.clamp(logits, -20, 20)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        logp = dist.log_prob(action)
        action_np = action.item()
        mu_logp_np = logp.item()
        logits_np = logits.detach().cpu().numpy()[0]
        if return_params:
            return action_np, mu_logp_np, logits_np
        else:
            return action_np, mu_logp_np

    def _perform_combined_update(self, actor_loss, critic_loss, kl, entropy):
        self.optimzer.zero_grad()
        kl.backward(retain_graph=True)
        k = [p.grad.clone() if p.grad is not None else torch.zeros_like(p) for p in self.model.parameters()]
        self.optimzer.zero_grad()

        actor_loss.backward(retain_graph=True)
        g = [p.grad.clone() if p.grad is not None else torch.zeros_like(p) for p in self.model.parameters()]
        self.optimzer.zero_grad()

        k_dot_g = sum((kg * gg).sum() for kg, gg in zip(k, g))
        k_norm_sq = sum((kg * kg).sum() for kg in k) + 1e-8
        alpha = (k_dot_g - self.delta) / k_norm_sq
        alpha = torch.clamp(alpha, min=0.0, max=1.0) if k_dot_g > 0 else torch.tensor(0.0, device=device)

        entropy_loss = -self.entropy_scale * entropy
        unprojected_loss = critic_loss + entropy_loss
        unprojected_loss.backward()

        for p, gg, kg in zip(self.model.parameters(), g, k):
            if p.grad is not None:
                p.grad.copy_(p.grad + (gg - alpha * kg))

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        self.optimzer.step()

        return alpha.item()

    def train(self, replay_buffer, batch_size=256, on_policy=False):
        states, actions, rewards, not_dones, mu_logps, next_states, mu_logits = self._prepare_batch(
            replay_buffer, batch_size, on_policy)

        B, T = states.shape[0], states.shape[1]
        flat_states = states.reshape(B * T, *self.state_dim)
        flat_actions = actions.view(-1).long()

        q_vals_all_actions = self.model(flat_states)[1]  # Get Q(s, .)
        q_vals = q_vals_all_actions.gather(1, flat_actions.unsqueeze(-1)).squeeze(-1).view(B, T)

        # We need LIVE V for advantage and STABLE V for retrace targets
        v_vals = self._compute_v_values(flat_states, use_trust_model=False).view(B, T)
        with torch.no_grad():
            v_vals_stable = self._compute_v_values(flat_states, use_trust_model=True).view(B, T)

        # PI(a|s)
        pi_logits, _ = self.model(flat_states)
        pi_logits = torch.clamp(pi_logits, -20, 20)
        pi_dist = torch.distributions.Categorical(logits=pi_logits)
        pi_probs = pi_dist.probs
        entropy = pi_dist.entropy().mean()

        # mu(a|s)
        mu_logits_flat = mu_logits.view(-1, self.action_dim)
        mu_dist = torch.distributions.Categorical(logits=mu_logits_flat)
        mu_probs = mu_dist.probs

        rho_all = pi_probs / (mu_probs + 1e-8)
        v_flat = v_vals.view(-1)
        adv_all = q_vals_all_actions.detach() - v_flat.unsqueeze(-1)
        rho_excess = torch.clamp(rho_all - self.c_bar, min=0.0)

        bias_term = (pi_probs * rho_excess * adv_all).sum(dim=-1)

        policy_logp = self.model.log_prob(flat_states, flat_actions).view(B, T)
        rho, rho_bar, c = self._compute_importance_weights(policy_logp, mu_logps)

        v_last = self._compute_bootstrap_value_stable(next_states[:, -1], not_dones[:, -1])
        v_tp1 = torch.empty(B, T, device=device)
        v_tp1[:, :-1] = v_vals_stable[:, 1:]
        v_tp1[:, -1] = v_last

        target_q = self._compute_retrace_targets(rewards, not_dones, q_vals, v_tp1, c)

        critic_loss = self._critic_loss_function(q_vals, target_q)

        pi_logits_seq = pi_logits.view(B, T, -1)
        actor_loss = 0.0
        for t in range(T):
            advantage_t = (target_q[:, t] - v_vals[:, t]).detach()
            log_pi_t = F.log_softmax(pi_logits_seq[:, t], dim=-1)
            logp_a_t = log_pi_t.gather(1, actions[:, t].unsqueeze(-1)).squeeze(-1)
            pg_loss_t = -(rho_bar[:, t] * logp_a_t * advantage_t).mean()

            start = t * B
            end = (t + 1) * B
            bc_loss_t = -bias_term[start:end].mean()
            actor_loss = actor_loss + pg_loss_t + bc_loss_t
        actor_loss = actor_loss / T

        ref_logits, _ = self.trust_region_model(flat_states)
        ref_logits = torch.clamp(ref_logits, -20, 20).detach()
        kl_for_grad = self._compute_kl(pi_logits, ref_logits).view(B, T)
        kl_for_grad = (kl_for_grad * not_dones).mean()
        alp = self._perform_combined_update(actor_loss, 0.5 * critic_loss, kl_for_grad, entropy)

        if torch.rand(1).item() < 0.01:
            print(
                f"critic_loss={critic_loss.item()}, "
                f"actor_loss={actor_loss.item()}, "
                f"mean_rho={rho.mean().item()}, "
                f"mean_c={c.mean().item():.7f},"
                f"mean_kl={kl_for_grad.item()},"
                f"On Policy={on_policy}, "
                f"alpha={alp}, "
            )
