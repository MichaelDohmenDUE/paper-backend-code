import numpy as np
import torch
import wandb

from backend.AbstractHandlers.AbstractActionHandler import AbstractActionHandler
from backend.Utils.src.BatchTransitioner import TransitionFactory
from backend.Utils.src.EnviromentHandler import EnvironmentHandler
from backend.Utils.src.ReplayBuffer import ReplayBuffer


class DataCollectionProcessor:
    def __init__(self, env_handler: EnvironmentHandler, transition_factory: TransitionFactory,
                 replay_buffer: ReplayBuffer, rollout_size: int,
                 action_handler: AbstractActionHandler):
        self.env_handler = env_handler
        self.transition_factory = transition_factory
        self.replay_buffer = replay_buffer
        self.rollout_size = rollout_size
        self.policy = action_handler
        self.total_steps = 0
        self.last_state = None

    def run(self):
        metrics = {
            "charts/episodic_return": 0,
            "charts/episodic_length": 0,
            "global_step": self.total_steps}
        self.replay_buffer.buffer.clear()
        episode_timesteps = 0
        episodic_reward = 0

        if self.last_state is None:
            self.last_state = self.env_handler.reset()
        state = self.last_state
        last_done = False
        for _ in range(self.rollout_size):
            self.total_steps += 1
            episode_timesteps += 1
            action, logp, value = self.policy.select_action(state)
            next_state, reward, done, info = self.env_handler.step(action)
            transition = self.transition_factory.forward(state=state, action=action, logp=logp, reward=reward,
                                                         done=done, value=value, bootstrap_value=None)
            self.replay_buffer.append(transition)
            episodic_reward += reward
            state = next_state
            last_done = done
            if done:
                metrics = {"charts/episodic_return": episodic_reward,
                    "charts/episodic_length": episode_timesteps,
                    "global_step": self.total_steps}
                state = self.env_handler.reset()
                episode_timesteps = 0
                episodic_reward = 0

        self.last_state = state
        if last_done:
            last_value = 0.0
        else:
            _, _, last_value = self.policy.select_action(state)
        self.replay_buffer.buffer[-1].bootstrap_value = last_value
        return metrics
