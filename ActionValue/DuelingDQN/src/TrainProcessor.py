import torch
from torch import nn

from backend.Utils.src.NodeLib.NodeLibrary import bellman, detransition, indexing, argmax, optimizer_normalized, \
    mean_squared_error
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
        states_tensor, actions_tensor, rewards_tensor, next_states_tensor, dones_tensor = detransition(
            self.buffer.spec.fields,
            batch,
            self.device)

        qs_behaviour = self.behavior_net(states_tensor)
        qsa_behavior = indexing(qs_behaviour, actions_tensor).reshape(-1)

        with torch.no_grad():
            next_actions = self.behavior_net(next_states_tensor)
            next_actions = argmax(next_actions)
            q_next = self.target_net(next_states_tensor)
            q_next = indexing(q_next, next_actions).reshape(-1)
            target = bellman(target_Q=q_next, reward=rewards_tensor, done=dones_tensor, discount_factor=self.gamma)

        loss = mean_squared_error(qsa_behavior, target)

        optimizer_normalized(self.behavior_net, self.optimizer, loss, self.max_grad_norm)

        metrics = {
            "losses/td_loss": loss.item(),
            "losses/q_values": qsa_behavior.mean().item(),
        }

        return metrics
