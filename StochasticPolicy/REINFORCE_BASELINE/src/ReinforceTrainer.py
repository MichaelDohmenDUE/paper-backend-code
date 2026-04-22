import torch


from backend.CommonModels.src.Policy_Reinforce_Baseline import PolicyReinforceBaseline
from backend.Utils.src import  RolloutBuffer
from backend.Utils.src.NodeLib.NodeLibrary import detransition, optimizer_update, policy_loss, mean_squared_error
from backend.Utils.src.utils import discounted_cumulative_reward
from backend.StochasticPolicy.REINFORCE_BASELINE.src import ActionHandler


class REINFORCETrainer:
    def __init__(self,
                 rollout_buffer: RolloutBuffer.RolloutBuffer,
                 behaviour: PolicyReinforceBaseline,
                 optimizer,
                 beta: float = 0.01,
                 gamma: float = 0.99,
                 device: torch.device = torch.device("cpu")
                 ):
        self.rollout_buffer = rollout_buffer
        self.behaviour = behaviour
        self.optimizer = optimizer
        self.beta = beta
        self.gamma = gamma
        self.device = device
        self.c_pol = 1.0
        self.c_val = 0.5

    def run(self):
        rollout = self.rollout_buffer.sample()
        state, logps, rewards, dones = detransition(self.rollout_buffer.spec.fields, rollout, self.device)
        _, value = self.behaviour(state)
        value = value.squeeze(-1)
        with torch.no_grad():
            G = discounted_cumulative_reward(self.gamma, rewards.cpu(), dones.cpu())
            G = torch.tensor(G, dtype=torch.float32).to(self.device)
            advantage = G - value.detach()
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        loss_policy = policy_loss(logps, advantage)

        loss_value = mean_squared_error(G, value)

        loss = loss_policy + 0.5 * loss_value

        optimizer_update(optimizer=self.optimizer, loss=loss)
        #Logging
        metrics = {
            "losses/policy_loss": loss_policy.item(),
            "losses / loss_vale": loss_value.item(),
            "losses / loss": loss.item(),
        }
        return metrics
