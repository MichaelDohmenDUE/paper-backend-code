import torch
import torch.nn.functional as F
from torch import nn

from backend.Utils.src.NodeLib.NodeLibrary import bellman
from backend.Utils.src.ReplayBuffer import ReplayBuffer


class TrainProcessor:
    def __init__(self, buffer: ReplayBuffer, behavior_net: nn.Module, target_net: nn.Module,
                 optimizer: torch.optim.Optimizer, gamma: float,
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

        states = batch["state"].to(self.device)
        actions = batch["action"].long().to(self.device)
        rewards = batch["reward"].to(self.device)
        next_states = batch["next_state"].to(self.device)
        dones = batch["done"].to(self.device)

        qsa_behavior = self.behavior_net(states).gather(1, actions)

        with torch.no_grad():
            next_actions = self.behavior_net(next_states).argmax(dim=1, keepdim=True)
            q_next = self.target_net(next_states).gather(1, next_actions)
            target = bellman(target_Q=q_next, reward=rewards, done=dones, discount_factor=self.gamma)

        loss = F.mse_loss(qsa_behavior, target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.behavior_net.parameters(), self.max_grad_norm) # TODO: I can't abstract here becuase of Clipping, think about the structure again
        self.optimizer.step()
