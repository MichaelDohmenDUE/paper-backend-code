import torch


from backend.CommonModels.src.Policy_Reinforce import PolicyVPG
from backend.Utils.src import ReplayBuffer, RolloutBuffer
from backend.Utils.src.NodeLib.NodeLibrary import detransition, optimizer_update, policy_loss, normalize
from backend.Utils.src.utils import discounted_cumulative_reward
from backend.StochasticPolicy.REINFORCE.src import ActionHandler


class REINFORCETrainer:
    def __init__(self,
                 rollout_buffer: RolloutBuffer.RolloutBuffer,
                 optimizer,
                 beta: float = 0.01,
                 gamma: float = 0.99,
                 device: torch.device = torch.device("cpu")
                 ):
        self.rollout_buffer = rollout_buffer
        self.optimizer = optimizer
        self.beta = beta
        self.gamma = gamma
        self.device = device

    def run(self):
        rollout = self.rollout_buffer.sample()
        logps, rewards, dones = detransition(self.rollout_buffer.spec.fields, rollout, self.device)
        with torch.no_grad():
            G = discounted_cumulative_reward(self.gamma, rewards.cpu(), dones.cpu())
            G = torch.tensor(G, dtype=torch.float32).to(self.device).view(-1)
        loss = policy_loss(logps, G)

        self.optimizer.zero_grad()
        loss.backward()

        total_squared_norm = 0.0
        n_params = 0
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    total_squared_norm += p.grad.detach().pow(2).sum().item()
                    n_params += p.numel()

        rms_grad = (total_squared_norm / n_params) ** 0.5 if n_params > 0 else 0
        self.optimizer.step()

        return {
            "grad/rms_policy": rms_grad,
            "grad/rms_gradient": rms_grad,
            "grad/total_parameters": n_params,
            "losses/policy_loss": loss.item()
        }
