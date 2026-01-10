import numpy as np
import torch
from torch import nn

from backend.AbstractHandlers.AbstractActionHandler import AbstractActionHandler


class ActionHandler(AbstractActionHandler):
    def __init__(self, actor: nn.Module, critic: nn.Module, device: torch.device):
        self.actor = actor
        self.critic = critic
        self.device = device

    def select_action(self, state):
        state = np.array(state, dtype=np.float32)
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            dist = self.actor(state_t)
            action = dist.sample()
            logp = dist.log_prob(action).sum(-1)
            value = self.critic(state_t).squeeze(-1)
        return action.cpu().numpy().squeeze(0), float(logp.cpu().numpy()), float(value.cpu().numpy())
