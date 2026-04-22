import numpy as np
import torch
import wandb
from torch import nn

from backend.ActionValue.DQN.src.ActionHandler import EpsilonGreedyPolicy
from backend.Utils.src import ReplayBuffer
from backend.Utils.src.BatchTransitioner import TransitionFactory
from backend.Utils.src.EnviromentHandler import EnvironmentHandler
from backend.Utils.src.NodeLib.NodeLibrary import reset_handler


class DataCollectionProcessor:
    def __init__(self, behaviour_net: nn.Module, env: EnvironmentHandler, buffer: ReplayBuffer,
                 eps_greedy: EpsilonGreedyPolicy, transition_factory: TransitionFactory, device: torch.device):
        self.behaviour_net = behaviour_net
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

    def run(self) -> None:
        with torch.no_grad():
            state_tensor = torch.tensor(self.state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.behaviour_net(state_tensor).squeeze(0)
        action = self.eps_greedy.forward(q_values=q_values)
        next_state, reward, done, done_bool = self.env.step(action)
        clipped_reward = np.sign(reward)
        transition = self.transition_factory.forward(state=self.state.astype(np.uint8), action=action,
                                                     reward=clipped_reward, next_state=next_state.astype(np.uint8),
                                                     done=done)

        self.buffer.append(transition)
        self.state = reset_handler(self.env, next_state, done)
        ###### Logging
        self.episode_reward += reward
        self.episode_steps += 1
        self.total_steps += 1
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
            self.episode_steps = 0