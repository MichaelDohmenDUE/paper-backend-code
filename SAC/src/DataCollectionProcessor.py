import torch
from torch import nn

from backend.DQN.src.ActionHandler import EpsilonGreedyPolicy
from backend.Utils.src.BatchTransitioner import TransitionFactory
from backend.Utils.src.EnviromentHandler import EnvironmentHandler
from backend.Utils.src.ReplayBuffer import ReplayBuffer


class DataCollectionProcessor:
    def __init__(self,env: EnvironmentHandler,policy,buffer: ReplayBuffer,transition_factory: TransitionFactory,device: torch.device):
        self.env = env
        self.policy = policy
        self.buffer = buffer
        self.transition_factory = transition_factory
        self.device = device

        self.state = self.env.reset()
        self.done = False
        self.episode_timesteps = 0

    def run(self):
        state_tensor = torch.as_tensor(self.state, dtype=torch.float32, device=self.device)

        action_tensor = self.policy.select_action(state_tensor)
        action_np = action_tensor.detach().cpu().numpy()

        self.episode_timesteps += 1
        next_state, reward, done, done_bool = self.env.step(
            action_np, episode_timesteps=self.episode_timesteps
        )

        transition = self.transition_factory.create(
            state=self.state,
            action=action_np,
            reward=reward,
            next_state=next_state,
            done=done_bool,
        )
        self.buffer.append(transition)

        self.state = next_state
        self.done = done_bool

        if self.done:
            self.state = self.env.reset()
            self.done = False
            self.episode_timesteps = 0

        return transition