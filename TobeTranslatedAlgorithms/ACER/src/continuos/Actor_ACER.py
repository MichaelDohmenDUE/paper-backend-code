import torch
import torch.nn as nn


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, -1.0, 2.0)
        std = torch.exp(log_std)
        return mean, std, log_std

    def _sample_raw(self, state):
        mean, std, log_std = self.forward(state)
        dist = torch.distributions.Normal(mean, std)
        pre_tanh = dist.rsample()
        action = torch.tanh(pre_tanh)
        log_prob = dist.log_prob(pre_tanh).sum(-1)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(-1)
        return action, log_prob, mean, log_std

    def sample_action(self, state):
        action, logp, _, _ = self._sample_raw(state)
        return action, logp

    def sample_action_with_params(self, state):
        return self._sample_raw(state)

    def log_prob(self, state, action):
        mean, std, log_std = self.forward(state)
        dist = torch.distributions.Normal(mean, std)
        pre_tanh = torch.atanh(torch.clamp(action, -0.999, 0.999))
        log_p = dist.log_prob(pre_tanh).sum(-1)
        log_p -= torch.log(1 - action.pow(2) + 1e-6).sum(-1)
        return log_p
