import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class Critic(nn.Module):
    def __init__(self,state_size: int,action_size: int, hidden_size: int):
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(state_size + action_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, state: Tensor, action: Tensor) -> Tensor:
        concat = torch.cat((state, action), dim=1)

        x = F.relu(self.fc1(concat))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
