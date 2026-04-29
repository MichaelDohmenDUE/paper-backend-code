import torch
from torchgen.native_function_generation import self_to_out_signature

from backend.StochasticPolicy.SAC.src.ActionSelector import ActionSelector
from backend.Utils.src.BatchTransitioner import TransitionFactory
from backend.Utils.src.EnviromentHandler import EnvironmentHandler
from backend.Utils.src.NodeLib.NodeLibrary import reset_handler
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
        self.episode_timesteps = 0
        self.episode_reward = 0
        self.total_steps = 0
        self.episode = 0

    def run(self):
        metrics = dict()
        state_tensor = torch.as_tensor(self.state, dtype=torch.float32, device=self.device)

        action_tensor = self.policy.select_action(state_tensor)
        action_np = action_tensor.detach().cpu().numpy()
        next_state, reward, terminated, truncated, info = self.env.env.step(action_np)

        transition = self.transition_factory.forward(state=self.state, action=action_np, reward=reward,
                                                     next_state=next_state, done=terminated)
        self.buffer.append(transition)

        self.state = reset_handler(self.env, next_state, terminated or truncated)

        self.episode_timesteps += 1
        self.episode_reward += reward
        self.total_steps += 1
        if terminated or truncated:
            self.episode += 1
            metrics = {
                "charts/episodic_return": self.episode_reward,
                "charts/episodic_length": self.episode_timesteps,
                "global_step": self.total_steps,
                "episode": self.episode,
            }
            self.episode_timesteps = 0
            self.episode_reward = 0

        return metrics
