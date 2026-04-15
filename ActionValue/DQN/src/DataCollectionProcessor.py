import numpy as np
import torch
from torch import nn

from backend.ActionValue.DQN.src.ActionHandler import EpsilonGreedyPolicy
from backend.Utils.src import ReplayBuffer
from backend.Utils.src.BatchTransitioner import TransitionFactory
from backend.Utils.src.EnviromentHandler import EnvironmentHandler


class DataCollectionProcessor:
    def __init__(self, policy: nn.Module, env: EnvironmentHandler, buffer: ReplayBuffer,
                 action_selector: EpsilonGreedyPolicy, transition_factory: TransitionFactory, device: torch.device):
        self.policy = policy
        self.env = env
        self.buffer = buffer
        self.state = env.reset()
        self.done = False
        self.action_selector = action_selector
        self.transition_factory = transition_factory
        self.device = device
        # Logging
        self.episode_count = 0
        self.episode_reward = 0
        self.total_steps = 0
        self.episode_steps = 0

    def run(self) -> None:
        if self.done:
            if self.episode_count % 1000 == 0:
                print(
                    f"Total Steps {self.total_steps / (10 ** 7)} Episode [{self.episode_count}] {self.episode_reward}")

            self.episode_count += 1
            self.episode_reward = 0.0
            self.state = self.env.reset()
            self.done = False
            self.episode_steps = 0
        with torch.no_grad():
            #TODO: Save this as uint8 not float, look at CleanrRL for inspiration
            state_tensor = torch.tensor(self.state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.policy(state_tensor).squeeze(0)
        action = self.action_selector.select_action(q_values=q_values)
        self.action_selector.update()
        next_state, reward, done, done_bool = self.env.step(action)
        reward = np.sign(reward)
        self.done = done
        self.episode_steps += 1
        transition = self.transition_factory.create(state=self.state, action=action, reward=reward,
                                                    next_state=next_state, done=self.done)

        self.buffer.append(transition)
        self.episode_reward += reward
        self.state = next_state
        self.total_steps += 1
