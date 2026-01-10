from torch import nn
from torch.distributions import Categorical


class DiscreteActorPPO(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, state):
        logits = self.net(state)
        return Categorical(logits=logits)
