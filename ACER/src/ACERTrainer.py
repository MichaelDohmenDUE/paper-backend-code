from torch import optim, nn
from backend.CommonModels.src.Actor_ACER import Actor
from backend.CommonModels.src.Critic import Critic
import torch
import copy
import numpy as np

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
        self.gamma = gamma
        self.tau = tau
        self.delta = trust_region_delta
        self.truncation_constant = 1.0
        self.retrace_lambda = 1.0

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action, mu_logp = self.actor.sample_action(state)
        return action.detach().cpu().numpy()[0], mu_logp.detach().cpu().numpy()[0]

    def train(self, replay_buffer, batch_size=256):
        batch = replay_buffer.sample(batch_size)
        states, actions, next_states, rewards, not_dones, mu_logps= zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.FloatTensor(np.array(actions)).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(device)
        not_dones = torch.FloatTensor(np.array(not_dones)).unsqueeze(1).to(device)


        with torch.no_grad():
            next_action, log_prob = self.actor.sample_action(next_states)
            q_next = self.critic(next_states, next_action)
            target_q = rewards + not_dones * self.gamma * q_next
            # TODO: replace one-step TD target  Retrace lambda target

        # Critic Update
        q_val = self.critic(states, actions)
        critic_loss = nn.MSELoss()(q_val, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ACtor Update
        action, _ = self.actor.sample_action(states)
        q_val = self.critic(states, action)
        actor_loss = -(q_val).mean()

        # TODO: compute policy_logp = self.actor.log_prob(states, actions)
        # compute rho = exp(policy_logp - mu_logp), rho_bar = clamp(rho, max=self.truncation_constant)
        # use rho_bar * advantage for policy gradient term
        # add bias correction term with actions sampled from policy

        # Trust region KL penalty
        policy_mean, policy_std = self.actor.forward(states)
        ref_mean, ref_std = self.trust_region_actor.forward(states)

        kl = torch.log(ref_std / policy_std) + (policy_std**2 + (policy_mean - ref_mean)**2) / (2.0 * ref_std**2) - 0.5
        kl = kl.sum(dim=1).mean()
        kl_penalty = torch.relu(kl - self.delta)

        actor_loss = actor_loss + kl_penalty

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        synchronize(self.actor, self.trust_region_actor, tau=self.tau)