import torch
import torch.nn.functional as F
from torch import nn

from backend.Utils.src.NodeLib.NodeLibrary import bellman
from backend.Utils.src.NodeLib.NodeLibrary import optimizer_update
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
        if len(self.buffer) < self.buffer.batch_size:
            return

        batch = self.buffer.sample_batch()

        states_tensor = batch["state"].to(self.device)
        actions_tensor = batch["action"].to(self.device).long()
        rewards_tensor = batch["reward"].to(self.device)
        next_states_tensor = batch["next_state"].to(self.device)
        dones_tensor = batch["done"].to(self.device)

        # Q(s,a) from behavior net
        q_values = self.behavior_net(states_tensor)
        qsa_behavior = q_values.gather(1, actions_tensor)

        # Double DQN target
        with torch.no_grad():
            next_q_online = self.behavior_net(next_states_tensor)
            next_actions = next_q_online.argmax(dim=1, keepdim=True)

            next_q_target = self.target_net(next_states_tensor)
            qsa_target = next_q_target.gather(1, next_actions)

            target = bellman(target_Q=qsa_target, reward=rewards_tensor, done=dones_tensor, discount_factor=self.gamma)

        loss = F.mse_loss(qsa_behavior, target)
        # Optimize
        optimizer_update(optimizer=self.optimizer, loss=loss)
