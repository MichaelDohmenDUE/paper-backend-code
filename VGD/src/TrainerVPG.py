
import torch
import numpy as np

from backend.CommonModels.src.Policy import Policy
from backend.CommonModels.src.Policy_VPG import PolicyVPG
from backend.Utils.src.utils import discounted_cumulative_reward


class VPGTrainer:
    def __init__(self,
                 policy: PolicyVPG,
                 optimizer,
                 beta: float =0.01,
                 gamma: float =0.99
                 ):
        self.policy = policy
        self.optimizer  =optimizer
        self.baseline_mean = 0.0
        self.beta = beta
        self.gamma = gamma

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        dist = self.policy.dist_categorical(state)
        action_dist = dist.sample()
        log_prob = dist.log_prob(action_dist).squeeze(0)
        return int(action_dist.item()), log_prob

    def train_episode(self, env_handler):
        obs = env_handler.reset()
        logps, rewards = [], []
        done = False
        t = 0

        while not done:
            action, log_prob = self.select_action(obs)
            logps.append(log_prob)
            obs, reward, done, _ = env_handler.step(action, t)
            rewards.append(reward)
            t += 1


        rewards = np.array(rewards, dtype=np.float64)
        dones = np.zeros_like(rewards, dtype=np.bool_)
        G = discounted_cumulative_reward(self.gamma, rewards, dones)
        G = torch.tensor(G, dtype=torch.float32)


        b = torch.full_like(G, self.baseline_mean)
        advantages = G - b


        loss = -(torch.stack(logps) * advantages).sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        ep_return = float(sum(rewards))
        self.baseline_mean = (1 - self.beta) * self.baseline_mean + self.beta * ep_return

        return ep_return, len(rewards)