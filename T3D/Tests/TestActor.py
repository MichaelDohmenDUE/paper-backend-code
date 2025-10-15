import unittest
import torch

from backend.T3D.src.Actor import Actor


class TestActor(unittest.TestCase):
    def test_forward(self):
        state_size = 2
        action_size = 2
        hidden_size = 64
        max_action = 2.0
        actor = Actor(state_size, action_size, max_action, hidden_size)

        dummy_input = torch.randn(5, state_size)
        output = actor(dummy_input)
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, (5, action_size))

if __name__ == '__main__':
    unittest.main()