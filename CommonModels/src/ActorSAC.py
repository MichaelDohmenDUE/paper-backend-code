from torch import nn
import torch


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
        std = log_std.exp()


        noise = torch.randn_like(mean)
        z = mean + std * noise
        action = torch.tanh(z) * self.max_action

        log_prob = std # TODO: Compute Log_prob

        return action, log_prob
