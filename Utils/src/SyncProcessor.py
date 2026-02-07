from backend.Utils.src.GlobalCounter import GlobalCounter
import torch
from torch import nn


class SyncProcessor:
    def __init__(self, from_net: nn.Module, to_net: nn.Module, tau: float, sync_freq: int, gl_counter = None):
        self.from_net = from_net
        self.to_net = to_net
        self.tau = tau
        self.local_counter = 0
        self.sync_freq = sync_freq
        self.gl_counter = gl_counter

    def run(self) -> None:
        counter = self.gl_counter.get() if self.gl_counter is not None else self.local_counter
        if counter % self.sync_freq == 0:
            if self.tau == 1.0:
                self.hard_sync()
            else:
                self.soft_sync()
        if self.gl_counter is None:
            self.local_counter += 1

    def hard_sync(self):
        self.to_net.load_state_dict(self.from_net.state_dict())

    def soft_sync(self):
        with torch.no_grad():
            for from_param, to_param in zip(self.from_net.parameters(), self.to_net.parameters()):
                to_param.data.copy_(self.tau * from_param.data + (1.0 - self.tau) * to_param.data)
