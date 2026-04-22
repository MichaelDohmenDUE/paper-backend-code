import torch
import torch.nn as nn


class PolicyReinforceBaseline(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )

        self.policy = nn.Linear(hidden_dim, output_dim)
        self.value = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        x = self.model(state)
        value = self.value(x)
        logits = self.policy(x)
        return logits, value

    def dist_categorical(self, x):
        x = self.model(x)
        logits = self.policy(x)
        return torch.distributions.Categorical(logits=logits)
