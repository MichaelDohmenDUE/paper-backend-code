import numpy as np
import torch

from backend.CommonModels.src.Policy_VPG import PolicyVPG
from backend.Utils.src import EnviromentHandler, ReplayBuffer
from backend.Utils.src.utils import discounted_cumulative_reward
from backend.VGD.src import ActionHandler


class VPGTrainer:
    def __init__(self,
                 policy: PolicyVPG,
                 replay_buffer:ReplayBuffer.ReplayBuffer,
                 action_handler: ActionHandler.ActionHandler,
                 optimizer,
                 beta: float = 0.01,
                 gamma: float = 0.99,
                 device: torch.device = torch.device("cpu")
                 ):
        self.policy = policy
        self.action_handler = action_handler
        self.replay_buffer = replay_buffer
        self.optimizer = optimizer
        self.beta = beta
        self.gamma = gamma
        self.device = device

    def run(self):
        transitions = list(self.replay_buffer.buffer)
        rewards = torch.tensor([t.reward for t in transitions], dtype=torch.float32, device=self.device)
        dones = torch.tensor([t.done for t in transitions], dtype=torch.bool, device=self.device)
        logps = torch.stack([t.logp for t in transitions]).to(self.device)

        rewards_np = rewards.detach().cpu().numpy()
        dones_np = dones.detach().cpu().numpy()

        G = discounted_cumulative_reward(self.gamma, rewards_np, dones_np)
        G = torch.tensor(G, dtype=torch.float32).to(self.device)

        loss = -(logps * G).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        ep_return = float(sum(rewards))

        return ep_return, len(rewards)
