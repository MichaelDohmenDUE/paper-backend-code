import torch.nn as nn


class DuellingDQN(nn.Module):
    """
    https://arxiv.org/pdf/1511.06581
    """

    def __init__(self, observation_size, hidden_size, action_size):
        super(DuellingDQN, self).__init__()
        self.common = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.ReLU()
        )
        self.state_value = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        self.advantage = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )

    def forward(self, x):
        x = self.common(x)
        value = self.state_value(x)
        advantage = self.advantage(x)
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values



from torch import nn


def preprocess(state):
    return state / 255.0


import torch
from torch import nn


def preprocess(state):
    return state / 255.0


class DuellingAtariDQN(nn.Module):
    def __init__(self, action_size):
        super().__init__()

        self.body = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        self.state_value = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        self.advantage_estimation = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, action_size)
        )

    def forward(self, x):
        x = preprocess(x)
        x = self.body(x)

        x = x.view(x.size(0), -1)

        value = self.state_value(x)
        advantage = self.advantage_estimation(x)

        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values