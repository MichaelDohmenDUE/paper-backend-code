import torch
from torch import nn

from backend.DQN.src.ActionHandler import EpsilonGreedyPolicy
from backend.Utils.src.BatchTransitioner import TransitionFactory
from backend.Utils.src.EnviromentHandler import EnvironmentHandler
from backend.Utils.src.ReplayBuffer import ReplayBuffer


class DataCollectionProcessor:
    def __init__(self, policy: nn.Module, env: EnvironmentHandler,
                 buffer: ReplayBuffer, action_selector: EpsilonGreedyPolicy,
                 transition_factory: TransitionFactory, device: torch.device):

        self.policy = policy
        self.env = env
        self.buffer = buffer
        self.action_selector = action_selector
        self.transition_factory = transition_factory
        self.device = device

        self.state = env.reset()
        self.done = False
        self.episode_reward = 0.0
        self.episode_count = 0
        self.total_steps = 0

    def run(self):
        if self.done:
            if self.episode_count % 10 == 0:
                print(f"Episode [{self.episode_count}] Reward: {self.episode_reward:.2f}")

            self.episode_count += 1
            self.episode_reward = 0.0
            self.state = self.env.reset()
            self.done = False

        with torch.no_grad():
            state_tensor = torch.as_tensor(self.state, dtype=torch.float32, device=self.device)
            q_values = self.policy(state_tensor)

        action = self.action_selector.select_action(q_values)
        next_state, reward, done, done_bool = self.env.step(action.item(), self.total_steps)

        transition = self.transition_factory.create(
            state=self.state,
            action=action.item(),
            reward=reward,
            next_state=next_state,
            done=done
        )

        self.buffer.append(transition)

        self.state = next_state
        self.done = done
        self.episode_reward += reward
        self.total_steps += 1
