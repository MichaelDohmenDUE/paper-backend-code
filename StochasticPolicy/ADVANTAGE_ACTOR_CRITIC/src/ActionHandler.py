import torch
from torch import nn

from backend.AbstractHandlers.AbstractActionHandler import AbstractActionHandler


class ActionHandler(AbstractActionHandler):
    def __init__(self, policy: nn.Module, device: torch.device):
        super().__init__()
        self.policy = policy
        self.device = device

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        dist = self.policy.dist_categorical(state)
        action_dist = dist.sample()
        log_prob = dist.log_prob(action_dist).squeeze(0)
        return int(action_dist.item()), log_prob