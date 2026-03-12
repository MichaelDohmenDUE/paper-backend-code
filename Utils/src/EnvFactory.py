from abc import ABC, abstractmethod

import gymnasium as gym

class EnvFactory(ABC):
    @abstractmethod
    def create(self):
        pass



class MujocoEnvFactory(EnvFactory):
    def __init__(self, env_name: str):
        self.env_name = env_name

    def create(self):
        return gym.make(self.env_name)


from gymnasium.wrappers import AtariPreprocessing, FrameStack

class AtariEnvFactory(EnvFactory):
    def __init__(self, env_name: str, frames=4):
        self.env_name = env_name
        self.frames = frames

    def create(self):
        env = gym.make(self.env_name, frameskip=1)

        env = AtariPreprocessing(
            env,grayscale_obs=True, scale_obs=True, screen_size=84,
            terminal_on_life_loss=False, noop_max=30, frame_skip=4
        )
        env = FrameStack(env, self.frames)
        return env
