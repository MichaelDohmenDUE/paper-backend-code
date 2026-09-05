import gymnasium as gym
import numpy as np

from Utils.src.EnvFactory import EnvFactory


class EnvironmentHandler:
    def __init__(self, factory: EnvFactory, seed: int, reward_scale: float = 1.0):
        self.env = factory.create()
        self.seed = seed
        self.env.reset(seed=seed)
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

    def step(self, action):
        next_state, reward, terminated, truncated, info = self.env.step(action)
        reward *= self.reward_scale
        # if self.episode_max_steps is not None and episode_timesteps >= self.episode_max_steps:
        # sv    truncated = True
        done = terminated or truncated

        next_state = np.asarray(next_state, dtype=np.float32)
        return next_state, reward, done, info

    def step_ddpg(self, action):
        next_state, reward, terminated, truncated, info = self.env.step(action)
        reward *= self.reward_scale

        next_state = np.asarray(next_state, dtype=np.float32)
        return next_state, reward, terminated, truncated, info

    def get_env_specs(self) -> tuple[int, int, float | None]:
        extracted_value = self.state_dim, self.action_dim, self.max_action
        return extracted_value


class VecEnvironmentHandler:
    def __init__(self, factory: EnvFactory, seed: int, num_envs: int = 8, reward_scale: float = 1.0, **factory_kwargs):
        self.num_envs = num_envs
        self.reward_scale = reward_scale
        self.seed = seed
        self.envs = gym.vector.SyncVectorEnv([
            lambda: factory.create(**factory_kwargs) for _ in range(num_envs)
        ])
        seeds = [seed + i for i in range(num_envs)]
        self.envs.reset(seed=seeds)
        obs_space = self.envs.single_observation_space
        self.state_dim = obs_space.shape

        action_space = self.envs.single_action_space
        if isinstance(action_space, gym.spaces.Discrete):
            self.action_dim = action_space.n
            self.max_action = None
        else:
            self.action_dim = action_space.shape[0]
            self.max_action = float(action_space.high[0])

    def reset(self):
        states, _ = self.envs.reset()
        return np.array(states, dtype=np.float32)

    def step(self, actions):
        next_states, rewards, terminated, truncated, infos = self.envs.step(actions)
        rewards = rewards * self.reward_scale
        dones = np.logical_or(terminated, truncated)
        next_states = np.asarray(next_states, dtype=np.float32)
        return next_states, rewards, dones, infos

    def step_detailed(self, actions):
        next_states, rewards, terminated, truncated, infos = self.envs.step(actions)
        rewards = rewards * self.reward_scale
        next_states = np.asarray(next_states, dtype=np.float32)
        return next_states, rewards, terminated, truncated, infos

    def get_env_specs(self) -> tuple:
        return self.state_dim, self.action_dim, self.max_action

    def close(self):
        self.envs.close()
