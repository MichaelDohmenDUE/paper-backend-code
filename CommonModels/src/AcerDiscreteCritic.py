import torch
import torch.nn as nn
import torch.nn.functional as F

class DiscreteCritic(nn.Module):
    def __init__(self, state_size: int, action_size: int, hidden_size: int):
        super().__init__()

        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)  # one Q per action

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)  # shape [batch, action_dim]
        return q_values

    def q_value(self, state, action):
        q_all = self.forward(state)
        return q_all.gather(1, action.long().unsqueeze(-1)).squeeze(-1)
