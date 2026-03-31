import gymnasium as gym
import numpy as np
import torch

from backend.Utils.src.EnvFactory import EnvFactory


class EnvironmentHandler:
    def __init__(self, factory: EnvFactory, seed: int, reward_scale: float = 1.0):
        self.env = factory.create()

        self.env.action_space.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        self.reward_scale: float = reward_scale

        self.reward_scale = reward_scale
        obs_space = self.env.observation_space

        if hasattr(obs_space, "shape") and len(obs_space.shape) > 1:
            # Image-based env (Atari)
            self.state_dim = obs_space.shape
        else:
            # Vector env (Acrobot, MuJoCo, CartPole)
            self.state_dim = obs_space.shape[0]

        if isinstance(self.env.action_space, gym.spaces.Discrete):
            self.action_dim: int = self.env.action_space.n
            self.max_action = None
        else:
            self.action_dim: int = self.env.action_space.shape[0]
            self.max_action = float(self.env.action_space.high[0])

        self.episode_max_steps = getattr(self.env, "_max_episode_steps", None) or getattr(self.env.spec,
                                                                                          "max_episode_steps", None)

    def reset(self):
        state, _ = self.env.reset()
        return np.array(state, dtype=np.float32)

    def step(self, action, episode_timesteps: int):
        next_state, reward, terminated, truncated, info = self.env.step(action)
        reward *= self.reward_scale
        if self.episode_max_steps is not None and episode_timesteps >= self.episode_max_steps:
            truncated = True
        done = terminated or truncated
        if truncated:
            done_bool = 0.0
        else:
            done_bool = float(terminated)

        next_state = np.asarray(next_state, dtype=np.float32)
        return next_state, reward, done, done_bool

    def get_env_specs(self) -> tuple[int, int, float | None]:
        extracted_value = self.state_dim, self.action_dim, self.max_action
        return extracted_value
