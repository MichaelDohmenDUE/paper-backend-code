import torch
from torch import nn
import torch.nn.functional as F
from backend.Utils.src.ReplayBuffer import ReplayBuffer


class TrainProcessor:
    def __init__(self,buffer: ReplayBuffer, behavior_net: nn.Module, target_net: nn.Module, optimizer: torch.optim.Optimizer, gamma: float,
                 device: torch.device, max_grad_norm: float = 10.0):
        self.buffer = buffer
        self.behavior_net = behavior_net
        self.target_net = target_net
        self.optimizer = optimizer
        self.gamma = gamma
        self.device = device
        self.max_grad_norm = max_grad_norm

    def run(self):
        if len(self.buffer) < self.buffer.batch_size:
            return

        batch = self.buffer.sample_batch()

        states      = batch["state"].to(self.device)
        actions     = batch["action"].long().to(self.device)
        rewards     = batch["reward"].to(self.device)
        next_states = batch["next_state"].to(self.device)
        dones       = batch["done"].to(self.device)

        qsa_behavior = self.behavior_net(states).gather(1, actions)

        with torch.no_grad():
            next_actions = self.behavior_net(next_states).argmax(dim=1, keepdim=True)
            q_next = self.target_net(next_states).gather(1, next_actions)
            target = rewards + self.gamma * q_next * (1.0 - dones)

        loss = F.smooth_l1_loss(qsa_behavior, target)

        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.behavior_net.parameters(), self.max_grad_norm)

        self.optimizer.step()
