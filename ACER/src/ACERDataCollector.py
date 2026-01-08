import torch

from backend.ACER.src.ACERTrainer import ACERTrainer
from backend.Utils.src.BatchTransitioner import TransitionFactory
from backend.Utils.src.EnviromentHandler import EnvironmentHandler
from backend.Utils.src.ReplayBuffer import ReplayBuffer


class ACERDataCollector:
    def __init__(self, trainer: ACERTrainer, env: EnvironmentHandler, buffer: ReplayBuffer, factory: TransitionFactory,
                 device: torch.device):
        self.trainer = trainer
        self.env = env
        self.buffer = buffer
        self.factory = factory
        self.device = device

        self.state = env.reset()
        self.done = False
        self.episode_reward = 0
        self.episode_timesteps = 0
        self.episode_count = 0

    def run(self):
        if self.done:
            print(f"Episode {self.episode_count} Reward: {self.episode_reward}")
            self.state = self.env.reset()
            self.done = False
            self.episode_reward = 0
            self.episode_timesteps = 0
            self.episode_count += 1

        self.episode_timesteps += 1

        action, mu_logp, mu_mean, mu_log_std = self.trainer.select_action(
            self.state, return_params=True
        )

        next_state, reward, done, done_bool = self.env.step(
            action, self.episode_timesteps
        )

        transition = self.factory.create(
            state=self.state,
            action=action,
            reward=reward,
            next_state=next_state,
            mask=1.0 - done_bool,
            mu_logp=mu_logp,
            mu_mean=mu_mean,
            mu_log_std=mu_log_std
        )

        self.buffer.append(transition)

        self.state = next_state
        self.done = done
        self.episode_reward += reward
