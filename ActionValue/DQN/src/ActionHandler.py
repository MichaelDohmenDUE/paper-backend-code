import random

import torch


class EpsilonGreedyPolicy:
    def __init__(self, epsilon: float):
        self.epsilon = epsilon

    def select_action(self, q_values: torch.Tensor) -> torch.Tensor:
        """ Epsilon-greedy policy, returns random action if random number is < epsilon, else greedy action
                Randomly samples from max actions if there is a tie.
             """
        actions = torch.arange(len(q_values)).to(q_values.device)
        max_q_value = torch.max(q_values).to(q_values.device)
        max_idx = (q_values == max_q_value).to(torch.int64)
        greedy_action = random.choice(actions[max_idx == 1])

        return random.choice(actions) if random.random() < self.epsilon else greedy_action
