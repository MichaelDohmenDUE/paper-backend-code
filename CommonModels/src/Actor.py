import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module): # TODO: Refactor this with Fatih's implementation
    def __init__(self, state_size:int , action_size: int, max_action: float, hidden_size:int):
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        self.max_action = max_action

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.max_action * torch.tanh(self.fc3(x))
