import torch
import torch.nn.functional as F
from torch import nn

from backend.Utils.src.NodeLib.NodeLibrary import bellman, mean_squared_error, detransition
from backend.Utils.src.NodeLib.NodeLibrary import indexing, nl_max, optimizer_normalized
from backend.Utils.src.ReplayBuffer import ReplayBuffer

class TrainProcessor:
    def __init__(self, buffer: ReplayBuffer, behavior_net: nn.Module, target_net: nn.Module,
                 optimizer: torch.optim.Optimizer, gamma: float, max_norm: float, warmup_steps: int,
                 device: torch.device):
        self.buffer = buffer
        self.behavior_net = behavior_net.to(device)
        self.target_net = target_net.to(device)
        self.optimizer = optimizer
        self.gamma = gamma
        self.device = device
        self.max_norm = max_norm
        self.warmup_steps = warmup_steps

    def run(self):
        if len(self.buffer) < self.warmup_steps:
            return
        batch = self.buffer.sample_batch()
        states_tensor, actions_tensor, rewards_tensor, next_states_tensor, dones_tensor = detransition(
            self.buffer.spec.fields,
            batch,
            self.device)

        qs_behaviour = self.behavior_net(states_tensor)
        qsa_behavior = indexing(qs_behaviour, actions_tensor).reshape(-1)

        with torch.no_grad():
            qs_target = self.target_net(next_states_tensor)
            qsa_target = nl_max(qs_target).reshape(-1)
            target = bellman(target_Q=qsa_target, reward=rewards_tensor, done=dones_tensor, discount_factor=self.gamma)
        loss = mean_squared_error(qsa_behavior, target)

        metrics = {
            "losses/td_loss": loss.item(),
            "losses/q_values": qsa_behavior.mean().item(),
        }

        optimizer_normalized(self.behavior_net, self.optimizer, loss, self.max_norm)
        return metrics
