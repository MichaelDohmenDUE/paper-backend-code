import torch

from backend.StochasticPolicy.SAC.src.ActionSelector import ActionSelector
from backend.Utils.src.BatchTransitioner import TransitionFactory
from backend.Utils.src.EnviromentHandler import EnvironmentHandler
from backend.Utils.src.ReplayBuffer import ReplayBuffer


class DataCollectionProcessor:
    def __init__(self, env: EnvironmentHandler, policy: ActionSelector, buffer: ReplayBuffer,
                 transition_factory: TransitionFactory, device: torch.device):
        self.env = env
        self.policy = policy
        self.buffer = buffer
        self.transition_factory = transition_factory
        self.device = device

        self.state = self.env.reset()
        self.done = False
        self.episode_timesteps = 0

    def run(self):
        self.done = False#TODO: Move self.done Setting into EndingHandler (align with drawings)
        state_tensor = torch.as_tensor(self.state, dtype=torch.float32, device=self.device)

        action_tensor = self.policy.select_action(state_tensor)
        action_np = action_tensor.detach().cpu().numpy()

        self.episode_timesteps += 1
        next_state, reward, terminated, truncated, info = self.env.env.step(action_np)

        transition = self.transition_factory.forward(state=self.state, action=action_np, reward=reward,
                                                     next_state=next_state, done=terminated)
        self.buffer.append(transition)

        self.state = next_state
        self.done = terminated or truncated
        if self.done:
            self.state = self.env.reset()
            self.episode_timesteps = 0

        return transition
