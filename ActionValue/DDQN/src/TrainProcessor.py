import torch
import torch.nn.functional as F
from torch import nn

from backend.Utils.src.NodeLib.NodeLibrary import bellman, detransition, indexing, argmax, mean_squared_error
from backend.Utils.src.NodeLib.NodeLibrary import optimizer_update
from backend.Utils.src.ReplayBuffer import ReplayBuffer


class TrainProcessor:
    def __init__(self, buffer: ReplayBuffer, behavior_net: nn.Module, target_net: nn.Module,
                 optimizer: torch.optim.Optimizer, gamma: float, device: torch.device):
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
        states_tensor, actions_tensor, rewards_tensor, next_states_tensor, dones_tensor = detransition(
            self.buffer.spec.fields,
            batch,
            self.device)

        q_values = self.behavior_net(states_tensor)
        qsa_behavior = indexing(q_values, actions_tensor).reshape(-1)

        # Double DQN target
        with torch.no_grad():
            next_q_online = self.behavior_net(next_states_tensor)
            next_actions = argmax(next_q_online)

            next_q_target = self.target_net(next_states_tensor)
            qsa_target = indexing(next_q_target, next_actions).reshape(-1)
            target = bellman(target_Q=qsa_target, reward=rewards_tensor, done=dones_tensor, discount_factor=self.gamma)

        loss = mean_squared_error(qsa_behavior, target)
        # Optimize
        optimizer_update(optimizer=self.optimizer, loss=loss)

        metrics = {
            "losses/td_loss": loss.item(),
            "losses/q_values": qsa_behavior.mean().item(),
        }

        return metrics

