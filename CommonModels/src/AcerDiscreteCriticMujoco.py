import torch
import torch.nn as nn
import torch.nn.functional as F

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=512):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
        )

        self.head = nn.Linear(hidden_size, action_dim)

    def forward(self, state):
        x = self.fc(state)
        return self.head(x)