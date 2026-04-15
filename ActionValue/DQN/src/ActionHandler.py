import random

import torch

class EpsilonGreedyPolicy:
    def __init__(self, epsilon_start=1.0, epsilon_final=0.05, epsilon_decay=1_000_000):
        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon_start
        self.total_steps = 0

    def update(self):
        fraction = min(self.total_steps / self.epsilon_decay, 1.0)
        self.epsilon = self.epsilon_start - fraction * (self.epsilon_start - self.epsilon_final)
        self.total_steps += 1

    def select_action(self, q_values: torch.Tensor):
        if random.random() < self.epsilon:
            return random.randrange(q_values.shape[0])
        else:
            return torch.argmax(q_values).item()
