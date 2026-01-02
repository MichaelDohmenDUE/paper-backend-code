import torch
from torch import nn
import torch.nn.functional as F
from backend.Utils.src.ReplayBuffer import ReplayBuffer


class TrainProcessor:
    def __init__(self, buffer: ReplayBuffer, behavior_net: nn.Module, target_net: nn.Module,
                 optimizer: torch.optim.Optimizer, gamma: float,  device: torch.device):
        self.buffer = buffer
        self.behavior_net = behavior_net.to(device)
        self.target_net = target_net.to(device)
        self.optimizer = optimizer
        self.gamma = gamma
        self.device = device

    def run(self):
        if len(self.buffer) < self.buffer.batch_size:
            return

        batch = self.buffer.sample_batch()

        states_tensor      = batch["state"].to(self.device)
        actions_tensor     = batch["action"].to(self.device)
        rewards_tensor     = batch["reward"].to(self.device)
        next_states_tensor = batch["next_state"].to(self.device)
        dones_tensor       = batch["done"].to(self.device)

        actions_tensor = actions_tensor.long()
        qsa_behavior = self.behavior_net(states_tensor).gather(1, actions_tensor)  # ^y

        with torch.no_grad():
            qs_target = self.target_net(next_states_tensor) # batch_size x action_dim
            qsa_target = qs_target.max(dim=1, keepdim=True).values
            target = rewards_tensor + self.gamma * qsa_target * (1.0 - dones_tensor)

        # ToDo: How to model dependent steps without forwarding anything
        self.optimizer.zero_grad()
        loss = F.mse_loss(qsa_behavior, target)
        loss.backward()
        self.optimizer.step()

