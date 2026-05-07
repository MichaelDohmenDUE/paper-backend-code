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
