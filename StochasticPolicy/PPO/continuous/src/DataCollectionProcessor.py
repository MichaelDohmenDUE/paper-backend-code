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
        np.random.seed(env_handler.seed)
        torch.manual_seed(env_handler.seed)

        # Logging for cleanRL
        self.total_steps = 0

    def run(self):
        self.replay_buffer.buffer.clear()
        episode_timesteps = 0
        episodic_reward = 0
        state = self.env_handler.reset()
        for step in range(self.rollout_size):
            self.total_steps += 1
            episode_timesteps += 1
            action, logp, value = self.policy.select_action(state)
            next_state, reward, done, done_bool = self.env_handler.step(action, episode_timesteps)
            transition = self.transition_factory.forward(state=state, action=action, logp=logp, reward=reward,
                                                         done=done_bool, value=value, bootstrap_value=None)
            episodic_reward += reward
            self.replay_buffer.append(transition)
            state = next_state

            if done:
                wandb.log({
                    "charts/episodic_return": episodic_reward,
                    "charts/episodic_length": episode_timesteps,
                    "global_step": self.total_steps,
                })
                state = self.env_handler.reset()
                episode_timesteps = 0
                episodic_reward = 0

        _, _, last_value = self.policy.select_action(state)
        self.replay_buffer.buffer[
            -1].bootstrap_value = last_value  # overwrite last Bootstrap value that gets later extracted for GAE
