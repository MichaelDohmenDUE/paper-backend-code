from abc import ABC, abstractmethod

import gymnasium as gym

class EnvFactory(ABC):
    @abstractmethod
    def create(self):
        pass



class GymEnvFactory(EnvFactory):
    def __init__(self, env_name: str):
        self.env_name = env_name

    def create(self):
        return gym.make(self.env_name)

from abc import ABC, abstractmethod
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStack


class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        actions = env.unwrapped.get_action_meanings()
        assert "FIRE" in actions, "FireResetEnv only works for Atari games with a FIRE action"
        self.fire_action = actions.index("FIRE")

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs, _, terminated, truncated, info = self.env.step(self.fire_action)
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)
        return obs, info

class AtariEnvFactory(EnvFactory):
    def __init__(self, env_name: str, frames=4):
        self.env_name = env_name
        self.frames = frames

    def create(self):
        env = gym.make(self.env_name, frameskip=1)

        env = AtariPreprocessing(
            env,
            grayscale_obs=True,
            scale_obs=True,
            screen_size=84,
            terminal_on_life_loss=False,
            noop_max=30,
            frame_skip=4
        )
        env = FireResetEnv(env)
        env = FrameStack(env, self.frames)
        return env
