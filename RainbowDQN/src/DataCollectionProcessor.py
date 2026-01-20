import torch
from torch import nn

from backend.DQN.src.ActionHandler import EpsilonGreedyPolicy
from backend.Utils.src.BatchTransitioner import TransitionFactory
from backend.Utils.src.EnviromentHandler import EnvironmentHandler
from backend.Utils.src.ReplayBuffer import ReplayBuffer
from backend.Utils.src.StepBuffer import StepBuffer


class DataCollectionProcessor:
    def __init__(self, policy: nn.Module, env: EnvironmentHandler, buffer: ReplayBuffer, step_buffer: StepBuffer,
                 action_selector: EpsilonGreedyPolicy, transition_factory: TransitionFactory, device: torch.device):
        self.policy = policy
        self.env = env
        self.step_buffer = step_buffer
        self.replay_buffer = buffer
        self.state = env.reset()
        self.done = False
        self.action_selector = action_selector
        self.transition_factory = transition_factory
        self.device = device

        self.episode_count = 0
        self.episode_reward = 0.0
        self.total_steps = 0
        self.episode_steps = 0

    def run(self):
        if self.done:
            if self.episode_count % 10 == 0:
                print(f"Episode [{self.episode_count}] {self.episode_reward}")

            for t in self.step_buffer.flush_transitions():
                self.replay_buffer.append(t)
            self.episode_count += 1
            self.episode_reward = 0.0
            self.state = self.env.reset()
            self.done = False
            self.episode_steps = 0
        with torch.no_grad():
            state_tensor = torch.tensor(self.state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.policy(state_tensor).squeeze(0)
        action = self.action_selector.select_action(q_values=q_values)
        next_state, reward, done, done_bool = self.env.step(action.item(), self.episode_steps)
        self.done = done
        self.episode_steps += 1
        transition = self.transition_factory.create(state=self.state, action=action.item(), reward=reward,
                                                    next_state=next_state, done=self.done)

        n_step_transition = self.step_buffer.push(transition)
        if n_step_transition is not None:
            self.replay_buffer.append(n_step_transition)
        self.episode_reward += reward
        self.state = next_state
        self.total_steps += 1
