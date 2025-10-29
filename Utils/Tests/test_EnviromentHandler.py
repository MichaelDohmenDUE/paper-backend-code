import unittest
import numpy as np

from backend.Utils.src.EnviromentHandler import EnvironmentHandler


class TestEnvironmentHandler(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env_handler = EnvironmentHandler(env_name="HalfCheetah-v5", seed=0)

    def test_reset(self):
        state = self.env_handler.reset()
        self.assertIsInstance(state, np.ndarray)
        self.assertEqual(state.shape[0], self.env_handler.state_dim)

    def test_step(self):
        self.env_handler.reset()
        action = self.env_handler.env.action_space.sample()
        next_state, reward, done, done_bool = self.env_handler.step(action, episode_timesteps=1)

        self.assertIsInstance(next_state, np.ndarray)
        self.assertEqual(next_state.shape[0], self.env_handler.state_dim)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)
        self.assertIsInstance(done_bool, float)

    def test_discrete_env_init(self):
        env = EnvironmentHandler(env_name="CartPole-v1", seed=0)
        self.assertEqual(env.action_dim, env.env.action_space.n)
        self.assertIsNone(env.max_action)

    def test_continuous_env_init(self):
        env = EnvironmentHandler(env_name="HalfCheetah-v5", seed=0)
        self.assertEqual(env.action_dim, env.env.action_space.shape[0])
        self.assertEqual(env.max_action, float(env.env.action_space.high[0]))

if __name__ == "__main__":
    unittest.main()