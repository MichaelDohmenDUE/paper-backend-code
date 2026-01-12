import torch
from torch.optim import Optimizer


class KFACOptimizer(Optimizer): #TODO: implement or import this, importing is more difficult than expected on a modern python system
    def __init__(self, model, lr=1e-3):
        defaults = dict(lr=lr)
        super().__init__(model.parameters(), defaults)

        self.model = model
        self.lr = lr
        self.acc_stats = False
        self.accumulate = False
        self.steps = 0

    def zero_grad(self, set_to_none=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        p.grad.detach_()
                        p.grad.zero_()

    @torch.no_grad()
    def step(self, closure=None):
        self.steps += 1

        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.add_(p.grad, alpha=-lr)

        return None
