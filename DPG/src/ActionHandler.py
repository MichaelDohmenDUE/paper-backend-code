import numpy as np
import torch
from torch import nn


class ActionHandler:
    def __init__(self, actor: nn.Module, action_size: int, max_action: float | None, expl_noise: float,
                 device: torch.device):
        self.actor = actor
        if max_action is None:
            self.max_action = 1.0
        else:
            self.max_action = max_action
        self.expl_noise = expl_noise
        self.device = device
        self.action_size = action_size

    def select_action(self, state):
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device).reshape(1, -1)
        with torch.no_grad():
            action = self.actor(state).cpu().numpy().flatten()
        noise = np.random.normal(0, self.expl_noise, size=self.action_size)
        action = np.clip(action + noise, -self.max_action, self.max_action)

        return torch.as_tensor(action, dtype=torch.float32, device=self.device)
