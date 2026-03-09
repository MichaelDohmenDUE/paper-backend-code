import torch
import torch.nn as nn

import torch
import torch.nn as nn
from torch.distributions import Categorical


class DiscreteActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.logits_layer = nn.Linear(hidden_dim, action_dim)

        nn.init.orthogonal_(self.fc1.weight, gain=1.0)
        nn.init.orthogonal_(self.fc2.weight, gain=1.0)
        nn.init.orthogonal_(self.logits_layer.weight, gain=0.01)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        logits = self.logits_layer(x)
        return logits

    def sample_action(self, state):
        logits = self.forward(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        logp = dist.log_prob(action)
        return action, logp

    def sample_action_with_params(self, state):
        logits = self.forward(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        logp = dist.log_prob(action)
        return action, logp, logits

    def log_prob(self, state, action):
        logits = self.forward(state)
        dist = Categorical(logits=logits)
        return dist.log_prob(action)
