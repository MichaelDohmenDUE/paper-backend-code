import unittest
import torch
from backend.CommonModels.src.Critic import Critic


class TestCritic(unittest.TestCase):
    def test_forward(self):
        state_size = 1
        action_size = 1
        hidden_size = 64
        batch_size = 6

        critic = Critic(state_size, action_size, hidden_size)

        state = torch.randn(batch_size, state_size)
        action = torch.randn(batch_size, action_size)

        output = critic(state, action)

        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, (batch_size, 1))

if __name__ == '__main__':
    unittest.main()