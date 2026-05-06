import numpy as np
import torch
from torch import nn

from Utils.src.NodeLib.Node import Graph, PropsNode
from backend.Utils.src import RolloutBuffer
from backend.Utils.src.BatchTransitioner import TransitionFactory
from backend.Utils.src.EnviromentHandler import VecEnvironmentHandler
from backend.Utils.src.NodeLib.NodeLibrary import categorical_distribution, sample_distribution, \
    to_numpy_array, TransitionNode, BufferAppendingNode, to_tensor


class DataCollectionProcessor:
    def __init__(self, env_handler: VecEnvironmentHandler, transition_factory: TransitionFactory,
                 rollout_buffer: RolloutBuffer.RolloutBuffer, behaviour: nn.Module, device: torch.device):
        self.env_handler = env_handler
        self.episode_timesteps = 0
        self.total_steps = 0
        self.episode_reward = 0

        self.context = {
            "state": np.array(self.env_handler.reset()).astype(np.float32),
            "behaviour": behaviour,
            "env_handler": env_handler,
            "transition_factory": transition_factory,
            "buffer": rollout_buffer,
            "device": device,
            "num_envs": self.env_handler.num_envs,
        }

        nodes = [
            PropsNode("FormatToTensor", ["state", "device"], ["state_t"],
                      function=lambda s, d: to_tensor(s, d).unsqueeze(0), no_grad=True),

            PropsNode("PolicyNet", ["state_t"], ["logits", "value"], props=["behaviour"],
                      function=lambda net, s: net(s), no_grad=False),

            PropsNode("CreateDist", ["logits"], ["dist"],
                      function=categorical_distribution, no_grad=False),

            PropsNode("SampleAction", ["dist"], ["action_raw", "log_prob"],
                      function=sample_distribution, no_grad=False),

            PropsNode("FormatToArray", ["action_raw"], ["action"],
                      function=to_numpy_array, no_grad=True),

            PropsNode("EnvStep", ["action"],
                      ["next_state", "reward", "done", "info"], props=["env_handler"],
                      function=lambda env, a: env.step(a), no_grad=True),

            TransitionNode(
                factory=transition_factory,
                input_mapping={
                    "state": "state",
                    "logp": "log_prob",
                    "reward": "reward",
                    "done": "done",
                }
            ),

            BufferAppendingNode(),

            PropsNode("StateUpdate", ["next_state", "_buffer_updated"], ["state"],
                      function=lambda ns, signal: np.array(ns).astype(np.float32)),
        ]

        self.graph = Graph(nodes, initial_keys=list(self.context.keys()))

    def run(self):
        self.graph.run(self.context)

        reward = self.context["reward"]
        done = self.context["done"]

        self.episode_reward += reward
        self.episode_timesteps += 1
        self.total_steps += 1

        metrics = {}
        if done:
            metrics = {
                "charts/episodic_return": self.episode_reward,
                "charts/episodic_length": self.episode_timesteps,
                "global_step": self.total_steps,
            }
            # Reset local counters
            self.episode_reward = 0
            self.episode_timesteps = 0

        return metrics
