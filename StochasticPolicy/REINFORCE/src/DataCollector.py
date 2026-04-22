import torch
import wandb
from torch import nn

from backend.Utils.src import RolloutBuffer
from backend.Utils.src.BatchTransitioner import TransitionFactory
from backend.Utils.src.EnviromentHandler import EnvironmentHandler
from backend.Utils.src.NodeLib.NodeLibrary import reset_handler, categorical_distribution, sample_distribution


class DataCollectionProcessor:
    def __init__(self, env_handler: EnvironmentHandler, transition_factory: TransitionFactory,
                 rollout_buffer: RolloutBuffer.RolloutBuffer, behaviour: nn.Module, device: torch.device):
        self.env_handler = env_handler
        self.transition_factory = transition_factory
        self.rollout_buffer = rollout_buffer
        self.behaviour = behaviour
        self.state = self.env_handler.reset()
        self.episode_timesteps = 0
        self.total_steps = 0
        self.episode_reward = 0
        self.device = device

    def run(self):
        done = False
        while not done:
            state = torch.tensor(self.state, dtype=torch.float32).unsqueeze(0).to(self.device)
            logits = self.behaviour(state)
            dist = categorical_distribution(logits)
            action, log_prob = sample_distribution(dist)
            next_state, reward, done, _ = self.env_handler.step(action)
            transition = self.transition_factory.forward(logp=log_prob, reward=reward, done=done)
            self.rollout_buffer.append(transition)
            self.state = reset_handler(self.env_handler, next_state, done)
            # Logging
            self.episode_timesteps += 1
            self.total_steps += 1
            self.episode_reward += reward

            if done:
                try:
                    wandb.log({
                        "charts/episodic_return": self.episode_reward,
                        "charts/episodic_length": self.episode_timesteps,
                        "global_step": self.total_steps,
                    })
                except Exception as e:
                    print(f"Logging error: {e}")
                self.episode_timesteps = 0
                self.episode_reward = 0