import numpy as np
import torch
import wandb
from torch import nn

from backend.Utils.src.EnviromentHandler import VecEnvironmentHandler
from NodeLib.Node import Node, Graph, PropsNode
from backend.ActionValue.DQN.src.ActionHandler import EpsilonGreedyPolicy
from backend.Utils.src import ReplayBuffer
from backend.Utils.src.BatchTransitioner import TransitionFactory
from backend.Utils.src.NodeLib.NodeLibrary import  to_tensor, to_numpy_array, TransitionNode, BufferAppendingNode


class DataCollectionProcessor:
    def __init__(self, behaviour_net: nn.Module, env: VecEnvironmentHandler , buffer: ReplayBuffer,
                 eps_greedy: EpsilonGreedyPolicy, transition_factory: TransitionFactory, device: torch.device):
        self.behaviour_net = behaviour_net
        self.env = env
        self.buffer = buffer
        self.state = env.reset()
        self.done = False
        self.epsilon_greedy = eps_greedy
        self.transition_factory = transition_factory
        self.device = device
        self.running_rewards = np.zeros(self.env.num_envs)
        self.running_lengths = np.zeros(self.env.num_envs)
        self.total_steps = 0

        self.context = {
            "state": np.array(env.reset()).astype(np.uint8),
            "behaviour_net": behaviour_net,
            "env": env,
            "buffer": buffer,
            "epsilon_greedy": eps_greedy,
            "transition_factory": transition_factory,
            "device": device,
            "num_envs": self.env.num_envs,
        }

        nodes = [
            PropsNode("FormatToTensor", ["state", "device"], ["state_t"],
                      function=to_tensor, no_grad=True),
            PropsNode("CastFloat", ["state_t"], ["state_net_input"],
                      function=lambda s: s.float(), no_grad=True),
            PropsNode("BehaviourNet", ["state_net_input"], ["q_values"], props=["behaviour_net"],
                      function=lambda net, s: net(s), no_grad=True),
            PropsNode("EpsilonGreedy", ["q_values"], ["action_raw"], props=["epsilon_greedy"],
                      function=lambda epsilon_greed, q: epsilon_greed.forward(q_values=q), no_grad=True),
            PropsNode("FormatToArray", ["action_raw"], ["action"],
                      function=to_numpy_array, no_grad=True),
            PropsNode("EnvStep", ["action"], ["next_state_raw", "reward", "terminated", "truncated", "info"], props=["env"],
                      function=lambda env, a: env.step_detailed(a), no_grad=True),
            PropsNode("FormatNextState", ["next_state_raw"], ["next_state"],
                      function=lambda ns: np.array(ns).astype(np.uint8), no_grad=True),
            Node("CombineDones", ["terminated", "truncated"], ["done"],
                 function=lambda term, trunc: term | trunc, no_grad=True),
            Node("ExtractTrueNextState", ["next_state", "done", "info"], ["real_next_state"],
                 function=lambda ns, d, info: info.get("final_observation", ns) if (
                         d and isinstance(info, dict)) else ns, no_grad=True),
            TransitionNode(
                factory=transition_factory,
                input_mapping={
                    "state": "state",
                    "action": "action",
                    "reward": "reward",
                    "next_state": "real_next_state",
                    "done": "done"
                }
            ),
            BufferAppendingNode(),
            PropsNode("StateUpdate", ["next_state", "_buffer_updated"], ["state"],
                      function=lambda ns, signal: ns, no_grad=True),
        ]

        self.graph = Graph(nodes, initial_keys=list(self.context.keys()))


    def run(self):
        self.graph.run(self.context)

        rewards = np.atleast_1d(self.context["reward"])
        dones = np.atleast_1d(self.context["done"])

        self.running_rewards += rewards
        self.running_lengths += 1
        self.total_steps += self.env.num_envs

        combined_metrics = {}
        completed_returns = []
        completed_lengths = []

        for i in range(self.env.num_envs):
            if dones[i]:
                completed_returns.append(self.running_rewards[i])
                completed_lengths.append(self.running_lengths[i])
                self.running_rewards[i] = 0
                self.running_lengths[i] = 0

        if completed_returns:
            combined_metrics["charts/episodic_return"] = np.mean(completed_returns)
            combined_metrics["charts/episodic_length"] = np.mean(completed_lengths)

        return combined_metrics