import copy

import numpy as np
import torch
from torch import optim, nn

from backend.CommonModels.src.Actor_ACER import Actor
from backend.CommonModels.src.Critic import Critic
from backend.Utils.src.utils import synchronize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ACERTrainer:
    def __init__(self, state_size, action_size, hidden_size, learning_rate=3e-4, gamma=0.99, tau=0.005, trust_region_delta=0.01):
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
        self.retrace_lambda = 1.0
        self.seq_len = 10
        self.beta = 0.011

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action, mu_logp = self.actor.sample_action(state)
        return action.detach().cpu().numpy()[0], mu_logp.detach().cpu().numpy()[0]

    def train(self, replay_buffer, batch_size=256, on_policy=False):
        if on_policy:
            batch = [replay_buffer[-self.seq_len:]]  # wrap in list for consistency
        else:
            batch = replay_buffer.sample_sequence(self.seq_len, batch_size)

        states, actions, rewards, not_dones, mu_logps, next_states = [], [], [], [], [], []
        for sequence in batch:
            s, a, ns, r, nd, ml = zip(*sequence)
            states.append(s)
            actions.append(a)
            next_states.append(ns)
            rewards.append(r)
            not_dones.append(nd)
            mu_logps.append(ml)

        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.FloatTensor(np.array(actions)).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        rewards = torch.FloatTensor(np.array(rewards)).to(device)
        not_dones = torch.FloatTensor(np.array(not_dones)).to(device)
        mu_logps = torch.FloatTensor(np.array(mu_logps)).to(device)

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
            v_last = self.critic(next_states[:, -1, :], pi_last)

        v_tp1 = torch.empty(B, T, device=device)
        v_tp1[:, :-1] = v_vals[:, 1:]
        v_tp1[:, -1] = v_last.squeeze(-1)

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

        # Actor update (last timestep)
        s_last = states[:, -1, :]
        a_last = actions[:, -1, :]
        q_last = self.critic(s_last, a_last).squeeze(-1)
        with torch.no_grad():
            pi_action, _ = self.actor.sample_action(s_last)
            v_val = self.critic(s_last, pi_action).squeeze(-1)
        advantage = (q_last - v_val).detach()

        policy_logp_last = self.actor.log_prob(s_last, a_last)
        if on_policy:
            pg_loss = -(advantage).mean()
            bias_correction = torch.tensor(0.0, device=device)
        else:
            mu_logp_last = mu_logps[:, -1]
            rho_last = torch.exp(policy_logp_last - mu_logp_last)
            rho_bar_last = torch.clamp(rho_last, max=self.truncation_constant)
            pg_loss = -(rho_bar_last * advantage).mean()
            with torch.no_grad():
                policy_action_bc, _ = self.actor.sample_action(s_last)
                q_pi_bc = self.critic(s_last, policy_action_bc).squeeze(-1)
                bias_adv = (q_pi_bc - v_val).detach()
            bias_correction = -(torch.relu(rho_last - self.truncation_constant) * bias_adv).mean()

        actor_loss = pg_loss + bias_correction

        # Trust region KL penalty
        policy_mean, policy_std = self.actor.forward(s_last)
        ref_mean, ref_std = self.trust_region_actor.forward(s_last)
        kl = torch.log(ref_std / policy_std) + (policy_std ** 2 + (policy_mean - ref_mean) ** 2) / (
                    2.0 * ref_std ** 2) - 0.5
        kl = kl.sum(dim=1).mean()
        actor_loss += torch.relu(kl - self.delta)

        # Entropy regularization
        dist = torch.distributions.Normal(policy_mean, policy_std)
        entropy = dist.entropy().sum(dim=-1).mean()
        actor_loss += -self.beta * entropy

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        synchronize(self.actor, self.trust_region_actor, tau=self.tau)