import torch
import torch.nn as nn


class PolicyReinforceBaseline(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.policy = nn.Linear(hidden_dim, output_dim)
        self.value = nn.Linear(hidden_dim, 1)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(module.bias, 0)
        nn.init.orthogonal_(self.policy.weight, gain=0.01)
        nn.init.orthogonal_(self.value.weight, gain=1.0)

    def forward(self, state):
        x = self.model(state)
        value = self.value(x)
        logits = self.policy(x)
        return logits, value

    def dist_categorical(self, x):
        x = self.model(x)
        logits = self.policy(x)
        return torch.distributions.Categorical(logits=logits)
