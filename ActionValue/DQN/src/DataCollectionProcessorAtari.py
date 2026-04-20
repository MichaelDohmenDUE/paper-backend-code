import numpy as np
import torch
import wandb
from torch import nn

from backend.ActionValue.DQN.src.ActionHandler import EpsilonGreedyPolicy
from backend.Utils.src import ReplayBuffer
from backend.Utils.src.BatchTransitioner import TransitionFactory
from backend.Utils.src.EnviromentHandler import EnvironmentHandler


class DataCollectionProcessor:
    def __init__(self, policy: nn.Module, env: EnvironmentHandler, buffer: ReplayBuffer,
                 eps_greedy: EpsilonGreedyPolicy, transition_factory: TransitionFactory, device: torch.device):
        self.policy = policy
        self.env = env
        self.buffer = buffer
        self.state = env.reset()
        self.done = False
        self.eps_greedy = eps_greedy
        self.transition_factory = transition_factory
        self.device = device
        # Logging
        self.episode_count = 0
        self.episode_reward = 0
        self.total_steps = 0
        self.episode_steps = 0

    def reset_handler(self, next_state, done):
        self.state = next_state
        if done:
            try:
                wandb.log({
                    "charts/episodic_return": self.episode_reward,
                    "charts/episodic_length": self.episode_steps,
                    "global_step": self.total_steps,
                })
            except Exception as e:
                print(f"Logging error: {e}")
            self.episode_count += 1
            self.episode_reward = 0.0
            self.state = self.env.reset()
            self.done = False
            self.episode_steps = 0

    def run(self) -> None:
        with torch.no_grad():
            state_tensor = torch.tensor(self.state, device=self.device).unsqueeze(0).float()
            q_values = self.policy(state_tensor).squeeze(0)
        action = self.eps_greedy.select_action(q_values=q_values)
        self.eps_greedy.update()
        next_state, reward, done, done_bool = self.env.step(action)
        self.episode_reward += reward
        clipped_reward = np.sign(reward)
        self.done = done
        self.episode_steps += 1
        transition = self.transition_factory.forward(state=self.state.astype(np.uint8), action=action,
                                                     reward=clipped_reward, next_state=next_state.astype(np.uint8),
                                                     done=self.done)

        self.buffer.append(transition)
        self.reset_handler(next_state, done)
        self.total_steps += 1
