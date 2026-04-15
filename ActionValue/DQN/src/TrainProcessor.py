from backend.Utils.src.NodeLib.NodeLibrary import optimizer_update
from backend.Utils.src.NodeLib.NodeLibrary import bellman
import torch
import torch.nn.functional as F
from torch import nn

from backend.Utils.src.ReplayBuffer import ReplayBuffer


class TrainProcessor:
    """
    OLD TrainProcessor for algorithmic clarity, does not get used
    """

    def __init__(self, buffer: ReplayBuffer, behavior_net: nn.Module, target_net: nn.Module,
                 optimizer: torch.optim.Optimizer, gamma: float, device: torch.device):
        self.buffer = buffer
        self.behavior_net = behavior_net.to(device)
        self.target_net = target_net.to(device)
        self.optimizer = optimizer
        self.gamma = gamma
        self.device = device

    def run(self):
        if len(self.buffer) < 50000:
            return

        batch = self.buffer.sample_batch()

        states_tensor = batch["state"].to(self.device)
        actions_tensor = batch["action"].to(self.device)
        rewards_tensor = batch["reward"].to(self.device)
        next_states_tensor = batch["next_state"].to(self.device)
        dones_tensor = batch["done"].to(self.device)

        actions_tensor = actions_tensor.long()
        qsa_behavior = self.behavior_net(states_tensor).gather(1, actions_tensor)  # ^y

        with torch.no_grad():
            qs_target = self.target_net(next_states_tensor)  # batch_size x action_dim
            qsa_target = qs_target.max(dim=1, keepdim=True).values
            target = bellman(target_Q=qsa_target, reward=rewards_tensor, done=dones_tensor, discount_factor=self.gamma)
        loss = F.mse_loss(qsa_behavior, target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.behavior_net.parameters(), 5)
        self.optimizer.step()
        return loss
