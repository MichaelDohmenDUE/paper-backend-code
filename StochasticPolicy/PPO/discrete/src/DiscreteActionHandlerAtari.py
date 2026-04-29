import torch
from torch import nn

from backend.AbstractHandlers.AbstractActionHandler import AbstractActionHandler


class ActionHandler(AbstractActionHandler):
    def __init__(self, agent: nn.Module, device: torch.device):
        self.agent = agent
        self.device = device

    def select_action(self, state) -> tuple[int, float, float]:
        state_t = torch.as_tensor(state, dtype=torch.uint8, device=self.device)
        with torch.no_grad():
            dist, value = self.agent(state_t)
            action = dist.sample()
            logp = dist.log_prob(action)
        return action.cpu().numpy(), logp.cpu().numpy(), value.squeeze(-1).cpu().numpy()
