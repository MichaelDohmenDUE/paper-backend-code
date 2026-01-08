import torch
from torch import nn


class ActorSAC(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, max_action: float, hidden: int):
        super().__init__()
        self.max_action = max_action
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )

        self.mean = nn.Linear(hidden, act_dim)
        self.log_std = nn.Linear(hidden, act_dim)

    def forward(self, state):
        x = self.net(state)
        mean = self.mean(x)
        log_std = self.log_std(x)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        log_std = torch.clamp(log_std, -20, 2)
        std = log_std.exp()

        noise = torch.randn_like(mean)
        z = mean + std * noise

        raw_action = torch.tanh(z)
        action = raw_action * self.max_action

        # Gaussian log_prob
        log_prob = (-0.5 * ((z - mean) / (std + 1e-9)).pow(2) - log_std - 0.5 * torch.log(
            torch.tensor(2 * torch.pi, device=state.device))).sum(dim=-1, keepdim=True)

        # tanh correction
        log_prob -= torch.log(1 - raw_action.pow(2) + 1e-9).sum(dim=-1, keepdim=True)

        return action, log_prob
