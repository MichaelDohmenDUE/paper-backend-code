import torch


class OUNoise:
    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.2, device="cpu"):
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.device = device
        self.action_dim = action_dim
        self.reset()

    def reset(self):
        self.state = torch.zeros(self.action_dim, device=self.device)

    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * torch.randn(self.action_dim, device=self.device)
        self.state = self.state + dx
        return self.state


class DeterministicPolicyWithNoise:
    def __init__(self, actor, noise_process, max_action, device):
        self.actor = actor
        self.noise = noise_process
        self.max_action = max_action
        self.device = device

    def select_action(self, state_tensor):
        with torch.no_grad():
            action = self.actor(state_tensor)

        noise = self.noise.sample()
        action = action + noise
        action = action.clamp(-self.max_action, self.max_action)
        return action
