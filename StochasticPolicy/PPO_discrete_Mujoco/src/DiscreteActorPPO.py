import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class DiscreteActorPPO(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, action_dim), std=0.01),
        )

    def forward(self, state):
        logits = self.net(state)
        return Categorical(logits=logits)


class AtariPPOAgent(nn.Module):
    def __init__(self, action_dim, channels=4):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
        )

        self.actor_head = nn.Linear(512, action_dim)
        self.critic_head = nn.Linear(512, 1)

    def forward(self, x):
        x = x.float() / 255.0
        features = self.network(x)
        logits = self.actor_head(features)
        dist = Categorical(logits=logits)
        value = self.critic_head(features)
        return dist, value
