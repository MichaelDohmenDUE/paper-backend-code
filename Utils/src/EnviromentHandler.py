import torch
import numpy as np
import gymnasium as gym

class EnvironmentHandler:
    def __init__(self, env_name, seed: int):
        self.env = gym.make(env_name)
        self.env.action_space.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.max_action = float(self.env.action_space.high[0])
        self.episode_max_steps = getattr(self.env, "_max_episode_steps", None) or getattr(self.env.spec, "max_episode_steps", None)

    def reset(self):
        state, _ = self.env.reset()
        return state

    def step(self, action, episode_timesteps: int):
        next_state, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        done_bool = float(done) if episode_timesteps < self.episode_max_steps else 0.0
        return next_state, reward, done, done_bool