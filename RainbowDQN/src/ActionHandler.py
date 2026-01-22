import torch


class GreedyPolicy:
    @staticmethod
    def select_action(q_values: torch.Tensor) -> torch.Tensor:
        return q_values.argmax()
