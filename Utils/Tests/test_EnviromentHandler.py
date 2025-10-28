import unittest
import numpy as np

from backend.Utils.src.EnviromentHandler import EnvironmentHandler


class TestEnvironmentHandler(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.env_handler = EnvironmentHandler(env_name="HalfCheetah-v5", seed=0)

    def test_reset(self):
        state = self.env_handler.reset()
        self.assertIsInstance(state, np.ndarray)
        self.assertEqual(state.shape[0], self.env_handler.state_dim)

    def test_step(self):
        state = self.env_handler.reset()
        action = self.env_handler.env.action_space.sample()
        next_state, reward, done, done_bool = self.env_handler.step(action, episode_timesteps=1)

        self.assertIsInstance(next_state, np.ndarray)
        self.assertEqual(next_state.shape[0], self.env_handler.state_dim)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)
        self.assertIsInstance(done_bool, float)

if __name__ == "__main__":
    unittest.main()