import torch
import torch.nn as nn


class PolicyVPG(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, state):
        return self.model(state)

    def dist_categorical(self, x):
        logits = self.model(x)
        return torch.distributions.Categorical(logits=logits)
