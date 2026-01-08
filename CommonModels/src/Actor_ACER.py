import torch
import torch.nn as nn


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        mean = self.mean(x)
        std = torch.exp(self.log_std)
        return mean, std

    def _sample_raw(self, state):
        mean, std = self.forward(state)
        dist = torch.distributions.Normal(mean, std)
        raw_action = dist.rsample()
        action = torch.tanh(raw_action)
        gaussian_logp = dist.log_prob(raw_action).sum(dim=-1)
        log_det_jacobian = torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1)
        log_prob = gaussian_logp - log_det_jacobian

        return action, log_prob, mean, torch.log(std)

    def sample_action(self, state):
        action, logp, _, _ = self._sample_raw(state)
        return action, logp

    def sample_action_with_params(self, state):
        return self._sample_raw(state)

    def log_prob(self, state, action):
        action = action.clamp(-0.999, 0.999)
        raw_action = 0.5 * torch.log((1 + action) / (1 - action + 1e-6))
        mean, std = self.forward(state)
        dist = torch.distributions.Normal(mean, std)
        gaussian_logp = dist.log_prob(raw_action).sum(dim=-1)
        log_det_jacobian = torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1)
        return gaussian_logp - log_det_jacobian
