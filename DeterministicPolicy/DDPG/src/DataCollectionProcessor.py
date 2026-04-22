import torch

from backend.Utils.src import ReplayBuffer, GlobalCounter
from backend.Utils.src.BatchTransitioner import TransitionFactory
from backend.Utils.src.EnviromentHandler import EnvironmentHandler
from backend.Utils.src.NodeLib.NodeLibrary import reset_handler


class DataCollectionProcessor:
    def __init__(self, env: EnvironmentHandler, policy, buffer: ReplayBuffer, transition_factory: TransitionFactory,
                 global_counter: GlobalCounter, device: torch.device):
        self.env = env
        self.policy = policy
        self.buffer = buffer
        self.transition_factory = transition_factory
        self.device = device
        self.global_counter = global_counter
        self.episode_reward = 0
        self.state = self.env.reset()
        self.episode_timesteps = 0
        self.episode_idx = 0

    def run(self):
        state_tensor = torch.as_tensor(self.state, dtype=torch.float32, device=self.device)

        action_tensor = self.policy.select_action(state_tensor)
        action_np = action_tensor.cpu().numpy()

        next_state, reward, terminated, truncated, _ = self.env.step_ddpg(action_np)
        self.episode_reward += reward

        transition = self.transition_factory.forward(state=self.state, action=action_np, reward=reward,
                                                     next_state=next_state, done=terminated)
        self.buffer.append(transition)

        self.state = reset_handler(env=self.env, next_state=next_state, done=terminated or truncated)
        #Reset PolicyNoise
        if terminated or truncated:
            self.policy.noise.reset()
        #Logging
        self.episode_timesteps += 1
        self.global_counter.set(self.global_counter.get() + 1)
        metrics = {}
        if terminated or truncated:
            metrics = {
                "charts/episodic_return": self.episode_reward,
                "charts/episodic_length": self.episode_timesteps,
                "global_step": self.global_counter,
            }
            self.episode_reward = 0
            self.episode_timesteps = 0
            self.episode_idx += 1
        return metrics
