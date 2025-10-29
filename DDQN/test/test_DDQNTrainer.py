import unittest
from unittest.mock import MagicMock, patch
import torch
import numpy as np
from backend.DDQN.src.DDQNTrainer import DDQNTrainer
from backend.CommonModels.src.Policy import Policy

class TestDDQNTrainer(unittest.TestCase):
    def setUp(self):
        self.state_dim = 4
        self.action_dim = 2
        self.hidden_size = 32
        self.batch_size = 8
        tau = 1.0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.behavior_policy = Policy(self.state_dim, self.action_dim, self.hidden_size).to(self.device)
        self.target_policy = Policy(self.state_dim, self.action_dim, self.hidden_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.behavior_policy.parameters())


        self.mock_buffer = MagicMock()
        self.mock_buffer.sample.return_value = [
            (
                np.random.randn(self.state_dim),
                np.random.randint(self.action_dim),
                np.random.randn(),
                np.random.randn(self.state_dim),
                np.random.choice([True, False])
            )
            for _ in range(self.batch_size)
        ]
        self.mock_buffer.__len__.return_value = self.batch_size + 1

        self.trainer = DDQNTrainer(
            env_handler=None,
            behavior_policy=self.behavior_policy,
            target_policy=self.target_policy,
            optimizer=self.optimizer,
            buffer=self.mock_buffer,
            gamma=0.99,
            batch_size=self.batch_size,
            update_freq=1,
            tau = tau,
            device=self.device
        )

    def test_optimize(self):
        initial_weights = [p.clone() for p in self.behavior_policy.parameters()]
        self.trainer._optimize()
        updated_weights = [p for p in self.behavior_policy.parameters()]
        self.assertFalse(all(torch.equal(i, u) for i, u in zip(initial_weights, updated_weights)))

    @patch("backend.DDQN.src.DDQNTrainer.synchronize")
    def test_train_triggers_optimize_sync(self, mock_sync):
        self.trainer.total_steps = 0
        tau = 1.0
        self.trainer.train()
        self.assertEqual(self.trainer.total_steps, 1)
        mock_sync.assert_called_once_with(self.behavior_policy, self.target_policy, tau=tau)

if __name__ == "__main__":
    unittest.main()