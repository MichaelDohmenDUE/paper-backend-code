import torch


from backend.CommonModels.src.Policy_Reinforce import PolicyVPG
from backend.Utils.src import ReplayBuffer, RolloutBuffer
from backend.Utils.src.NodeLib.NodeLibrary import detransition, optimizer_update, policy_loss
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
            G = torch.tensor(G, dtype=torch.float32).to(self.device)
        loss = policy_loss(logps, G)

        optimizer_update(optimizer=self.optimizer, loss=loss)
        #Logging
        metrics = {
            "losses/policy_loss": loss.item(),
        }
        return metrics
