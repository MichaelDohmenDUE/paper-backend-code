import unittest

import numpy as np
import torch

from StochasticPolicy.PPO_continuous.src.PPOTrainerProcessor import PPOTrainer


class TestPPOTrainer(unittest.TestCase):
    def setUp(self):
        self.state_dim = 4
        self.action_dim = 2
        self.hidden_dim = 16
        self.trainer = PPOTrainer(self.state_dim, self.action_dim, self.hidden_dim)

    def test_select_action(self):
        state = np.zeros(self.state_dim, dtype=np.float32)
        action, logp, value = self.trainer.select_action(state)

        self.assertIsInstance(action, np.ndarray)
        self.assertIsInstance(logp, float)
        self.assertIsInstance(value, float)
        self.assertEqual(action.shape, (self.action_dim,))

    def test_train(self):
        n = 32
        states = np.random.randn(n, self.state_dim).astype(np.float32)
        actions = np.random.randn(n, self.action_dim).astype(np.float32)
        old_logps = np.random.randn(n).astype(np.float32)
        advantages = np.random.randn(n).astype(np.float32)
        returns = np.random.randn(n).astype(np.float32)

        self.trainer.train(states, actions, old_logps, advantages, returns,
                           batch_size=8, epochs=1)

        for p in list(self.trainer.actor.parameters()) + list(self.trainer.critic.parameters()):
            self.assertTrue(p.requires_grad)
            self.assertIsInstance(p.data, torch.Tensor)


if __name__ == "__main__":
    unittest.main()
