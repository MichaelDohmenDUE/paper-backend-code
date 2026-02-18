import torch
from torch import nn

from backend.Utils.src.PrioReplayBuffer import PrioReplayBuffer


class TrainProcessor:
    def __init__(self, buffer: PrioReplayBuffer, behavior_net: nn.Module, target_net: nn.Module,
                 optimizer: torch.optim.Optimizer, gamma: float,
                 device: torch.device,
                 v_min: float, v_max: float, atoms_size: int, max_grad_norm: float = 10.0):
        self.buffer = buffer
        self.behavior_net = behavior_net
        self.target_net = target_net
        self.optimizer = optimizer
        self.gamma = gamma
        self.device = device
        self.max_grad_norm = max_grad_norm
        self.v_min = v_min
        self.v_max = v_max
        self.atoms_size = atoms_size
        self.delta_z = (v_max - v_min) / (atoms_size - 1)
        self.support = torch.linspace(v_min, v_max, atoms_size).to(device)

    def run(self):
        if len(self.buffer) < self.buffer.batch_size:
            return

        batch = self.buffer.sample_batch()
        tensors: dict = batch["BatchTensor"]
        indices: torch.Tensor = batch["Indices"]
        weight: torch.Tensor = batch["Weights"]
        weights = weight.to(self.device)

        states = tensors["state"].to(self.device)
        actions = tensors["action"].long().to(self.device)
        rewards = tensors["reward"].to(self.device)
        next_states = tensors["next_state"].to(self.device)
        dones = tensors["done"].to(self.device)

        logits = self.behavior_net(states)
        log_probs = torch.log_softmax(logits, dim=-1)

        actions_expanded = actions.unsqueeze(-1).expand(-1, 1, self.atoms_size)
        log_probs_a = log_probs.gather(1, actions_expanded).squeeze(1)

        with torch.no_grad():
            next_logits = self.behavior_net(next_states)
            next_probs = torch.softmax(next_logits, dim=-1)
            q_next = (next_probs * self.support).sum(dim=-1)
            next_actions = q_next.argmax(dim=1, keepdim=True)

            target_logits = self.target_net(next_states)
            target_probs = torch.softmax(target_logits, dim=-1)
            next_actions_expanded = next_actions.unsqueeze(-1).expand(-1, 1, self.atoms_size)
            target_dist = target_probs.gather(1, next_actions_expanded).squeeze(1)

            Tz = rewards + (1 - dones) * self.gamma * self.support.view(1, -1)
            Tz = Tz.clamp(self.v_min, self.v_max)

            b = (Tz - self.v_min) / self.delta_z
            l = b.floor().long()
            u = b.ceil().long()

            l = l.clamp(0, self.atoms_size - 1)
            u = u.clamp(0, self.atoms_size - 1)

            proj_dist = torch.zeros_like(target_dist)

            offset = (u.float() - b)
            proj_dist.scatter_add_(1, l, target_dist * offset)

            offset = (b - l.float())
            proj_dist.scatter_add_(1, u, target_dist * offset)

        proj_dist = proj_dist.clamp(min=1e-6)
        loss_per_sample = -(proj_dist * log_probs_a).sum(dim=-1)
        loss = (weights * loss_per_sample).mean()

        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.behavior_net.parameters(), self.max_grad_norm)

        self.optimizer.step()
        self.behavior_net.reset_noise()
        self.target_net.reset_noise()

        # Update Prios
        new_priorities = loss_per_sample.detach().abs().cpu().numpy().flatten()
        self.buffer.update_priorities(indices, new_priorities)
