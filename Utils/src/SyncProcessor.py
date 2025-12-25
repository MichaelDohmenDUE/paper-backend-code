from torch import nn

class SyncProcessor:
    def __init__(self, from_net: nn.Module, to_net: nn.Module, tau: float, sync_freq: int):
        self.from_net = from_net
        self.to_net = to_net
        self.tau = tau
        # Each process runs sequentially and can
        self.counter = 0
        self.sync_freq = sync_freq

    def run(self) -> None:
        if self.counter % self.sync_freq == 0:
            if self.tau == 1.0:
                self.hard_sync()
            else:
                self.soft_sync()
        self.counter +=1

    def hard_sync(self):
        self.to_net.load_state_dict(self.from_net.state_dict())

    def soft_sync(self):
        for from_param, to_param in zip(self.from_net.parameters(), self.to_net.parameters()):
            to_param.copy_(self.tau * from_param + (1.0 - self.tau) * to_param)
