import torch
from torch import nn


class ActionSelector(nn.Module):
    def __init__(self, actor_net: nn.Module, max_action: float ,device: torch.device):
        super().__init__()
        self.actor = actor_net
        self.device = device
        self.max_action = max_action

    def select_action(self, state_tensor):
        mean, log_std = self.actor(state_tensor)
        std = log_std.exp()

        noise = torch.randn_like(mean)
        z = mean + std * noise

        action = torch.tanh(z) * self.max_action
        return action
