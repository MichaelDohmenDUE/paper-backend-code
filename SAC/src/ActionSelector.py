import torch
from torch import nn


class ActionSelector(nn.Module):
    def __init__(self, actor_net: nn.Module, device: torch.device):
        super().__init__()
        self.actor = actor_net
        self.device = device
        self.max_action = max_action

    def select_action(self, state_tensor):
        action, log_std = self.actor.sample(state_tensor)
        return action
