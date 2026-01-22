import torch.nn as nn


class RainbowDuellingDQN(nn.Module):
    """
    https://arxiv.org/abs/1710.02298
    """

    def __init__(self, observation_size: int, hidden_size: int, action_size: int, atoms_size:int):
        self.atoms_size = atoms_size
        self.action_size = action_size
        super(RainbowDuellingDQN, self).__init__()
        self.common = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.ReLU()
        )
        self.state_value = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, atoms_size)
        )
        self.advantage = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size * atoms_size)
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.common(x)
        value = self.state_value(x).view(batch_size, 1, self.atoms_size)
        advantage = self.advantage(x).view(batch_size, self.action_size, self.atoms_size)
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values
