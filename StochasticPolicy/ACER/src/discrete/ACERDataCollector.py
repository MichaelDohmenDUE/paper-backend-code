import numpy as np
import torch

from backend.StochasticPolicy.ACER.src.discrete.ACERTrainer import ACERTrainer
from backend.Utils.src.BatchTransitioner import TransitionFactory
from backend.Utils.src.EnviromentHandler import EnvironmentHandler
from backend.Utils.src.ReplayBuffer import ReplayBuffer


class ACERDataCollector:
    def __init__(self, trainer: ACERTrainer, env: EnvironmentHandler, buffer: ReplayBuffer, factory: TransitionFactory,
                 device: torch.device, seq_len: int = 20):
        self.trainer = trainer
        self.env = env
        self.buffer = buffer
        self.factory = factory
        self.device = device
        self.seq_len = seq_len

        self.rollout = []
        self.state = np.array(env.reset(), dtype=np.float32)
        self.done = False
        self.episode_reward = 0
        self.episode_timesteps = 0
        self.episode_count = 0

    def run(self):
        if self.done:
            #print(f"Episode {self.episode_count} Reward: {self.episode_reward}")
            self.state = np.array(self.env.reset(), dtype=np.float32)
            self.done = False
            self.episode_reward = 0
            self.episode_timesteps = 0
            self.episode_count += 1

        self.episode_timesteps += 1

        action, mu_logp, logtis = self.trainer.select_action(
            self.state, return_params=True
        )

        next_state, reward, done, done_bool = self.env.step(action, self.episode_timesteps)
        next_state = np.array(next_state, dtype=np.float32)

        #slipped_reward = np.clip(reward, -1, 1)

        transition = self.factory.create(
            state=self.state,
            action=action,
            reward=reward,
            next_state=next_state,
            mask=1.0 - done_bool,
            mu_logp=mu_logp,
            mu_logits=logtis
        )

        self.rollout.append(transition)
        self.state = next_state
        self.done = done
        self.episode_reward += reward

        if len(self.rollout) == self.seq_len:
            for tr in self.rollout:
                #print(tr)
                self.buffer.append(tr)
            rollout = self.rollout
            self.rollout = []
            return rollout
