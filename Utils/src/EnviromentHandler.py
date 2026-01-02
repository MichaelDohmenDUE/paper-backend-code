import torch
import numpy as np
import gymnasium as gym

class EnvironmentHandler:
    def __init__(self, env_name: str, seed: int, reward_scale: float = 1.0):
        self.env = gym.make(env_name)
        self.env.action_space.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        self.reward_scale: float = reward_scale

        self.state_dim: int = self.env.observation_space.shape[0]

        if isinstance(self.env.action_space, gym.spaces.Discrete):
            self.action_dim: int = self.env.action_space.n
            self.max_action = None
        else:
            self.action_dim :int  = self.env.action_space.shape[0]
            self.max_action = float(self.env.action_space.high[0])

        self.episode_max_steps = getattr(self.env, "_max_episode_steps", None) or getattr(self.env.spec, "max_episode_steps", None)

    def reset(self):
        state, _ = self.env.reset()
        return state

    def step(self, action, episode_timesteps: int):
        next_state, reward, terminated, truncated, info = self.env.step(action)
        reward = reward * self.reward_scale
        done = terminated or truncated
        done_bool = float(done) if episode_timesteps < self.episode_max_steps else 0.0
        return next_state, reward, done, done_bool

    def get_env_specs(self)-> tuple[int, int ,float|None] :
        extracted_value = self.state_dim, self.action_dim, self.max_action
        return extracted_value
