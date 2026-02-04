import numpy as np
import torch
from torch import nn


class ActionHandler:
    def __init__(self, actor: nn.Module, action_size: int, max_action: float, expl_noise: float, noise_clip: float,
                 start_timesteps: int,
                 device: torch.device):
        self.actor = actor
        self.max_action = max_action
        self.expl_noise = expl_noise
        self.noise_clip = noise_clip
        self.start_timesteps = start_timesteps
        self.device = device
        self.action_size = action_size

    def select_action(self, state, t):
        if t < self.start_timesteps:
            return np.random.uniform(-self.max_action, self.max_action, self.action_size)

        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        with torch.no_grad():
            action = self.actor(state).cpu().numpy().flatten()
        noise = np.random.normal(0, self.max_action * self.expl_noise, size=self.action_size).clip(-self.noise_clip,
                                                                                                   self.noise_clip)
        return np.clip(action + noise, -self.max_action, self.max_action)
