import torch


class OUNoise:
    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.2, device="cpu"):
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.device = device
        self.action_dim = action_dim
        self.state = torch.zeros(self.action_dim, device=self.device)
        self.reset()

    def reset(self):
        self.state = torch.zeros(self.action_dim, device=self.device)

    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * torch.randn(self.action_dim, device=self.device)
        self.state = self.state + dx
        return self.state
