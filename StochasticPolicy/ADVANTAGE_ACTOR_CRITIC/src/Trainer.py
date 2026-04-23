import torch
from torch.distributions import Categorical

from backend.CommonModels.src.Policy_Reinforce_Baseline import PolicyReinforceBaseline
from backend.Utils.src import  RolloutBuffer
from backend.Utils.src.NodeLib.NodeLibrary import detransition, optimizer_update, policy_loss, mean_squared_error, \
    combined_loss, bellman, optimizer_normalized
from backend.Utils.src.utils import discounted_cumulative_reward


class Trainer:
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
        self.c_pol = 0.01
        self.c_val = 1.0

    def run(self):
        rollout = self.rollout_buffer.sample()
        state, logps, rewards, dones, next_state = detransition(self.rollout_buffer.spec.fields, rollout, self.device)
        logits, value = self.behaviour(state)
        _, next_value = self.behaviour(next_state)
        next_value = next_value.reshape(-1)
        value = value.reshape(-1)
        with torch.no_grad():
            G = bellman(next_value, rewards, dones, discount_factor=self.gamma)
            G = G.detach().clone().to(dtype=torch.float32, device=self.device)
        advantage = G - value.detach()
        loss_policy = policy_loss(logps, advantage)

        loss_value = mean_squared_error(value, G)
        # Entropy
        #dist = Categorical(logits=logits)
        #print(logps.shape)
        #entropy = dist.entropy().mean()
        #loss_entropy = -entropy

        loss = combined_loss(loss_policy, self.c_pol, loss_value, self.c_val)#, loss_entropy, self.beta)

        optimizer_update(optimizer=self.optimizer, loss=loss)
        #Logging
        metrics = {
            "losses/policy_loss": loss_policy.item(),
            "losses / loss_vale": loss_value.item(),
            "losses / loss": loss.item(),
            #"losses/entropy": entropy.item(),
        }
        return metrics
import torch
from torch.distributions import Categorical

from backend.CommonModels.src.Policy_Reinforce_Baseline import PolicyReinforceBaseline
from backend.Utils.src import  RolloutBuffer
from backend.Utils.src.NodeLib.NodeLibrary import detransition, optimizer_update, policy_loss, mean_squared_error, \
    combined_loss, bellman
from backend.Utils.src.utils import discounted_cumulative_reward


class Trainer:
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
        state, logps, rewards, dones, next_state = detransition(self.rollout_buffer.spec.fields, rollout, self.device)
        logits, value = self.behaviour(state)
        _, next_value = self.behaviour(next_state)
        next_value = next_value.reshape(-1)
        value = value.reshape(-1)
        with torch.no_grad():
            G = bellman(next_value, rewards, dones, discount_factor=self.gamma)
            G = G.detach().clone().to(dtype=torch.float32, device=self.device)
        advantage = G - value.detach()
        loss_policy = policy_loss(logps, advantage)

        loss_value = mean_squared_error(value, G)
        # Entropy
        #dist = Categorical(logits=logits)
        #print(logps.shape)
        #entropy = dist.entropy().mean()
        #loss_entropy = -entropy

        loss = combined_loss(loss_policy, self.c_pol, loss_value, self.c_val)#, loss_entropy, self.beta)

        optimizer_update(optimizer=self.optimizer, loss=loss)
        #Logging
        metrics = {
            "losses/policy_loss": loss_policy.item(),
            "losses / loss_vale": loss_value.item(),
            "losses / loss": loss.item(),
           # "losses/entropy": entropy.item(),
        }
        return metrics
