import torch
from torch import nn

from backend.Utils.src.PrioReplayBuffer import PrioReplayBuffer


class TrainProcessor:
    def __init__(self, buffer: PrioReplayBuffer, behavior_net: nn.Module, target_net: nn.Module,
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
        tensors = batch["BatchTensor"]
        indices = batch["Indices"]
        weights = batch["Weights"].to(self.device)

        states = tensors["state"].to(self.device)
        actions = tensors["action"].long().to(self.device)
        rewards = tensors["reward"].to(self.device)
        next_states = tensors["next_state"].to(self.device)
        dones = tensors["done"].to(self.device)

        qsa_behavior = self.behavior_net(states).gather(1, actions)

        with torch.no_grad():
            next_actions = self.behavior_net(next_states).argmax(dim=1, keepdim=True)
            q_next = self.target_net(next_states).gather(1, next_actions)
            target = rewards + self.gamma * q_next * (1.0 - dones)

        td_error = qsa_behavior - target
        loss = (weights.unsqueeze(1) * (td_error ** 2)).mean()

        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.behavior_net.parameters(), self.max_grad_norm)

        self.optimizer.step()
        # Update Prios
        new_priorities = td_error.detach().abs().cpu().numpy().flatten()
        self.buffer.update_priorities(indices, new_priorities)
