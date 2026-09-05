import numpy as np
import torch
from torch import nn

from ActionValue.DQN.src.EpsilonGreedy import EpsilonGreedyPolicy
from Utils.src import ReplayBuffer
from Utils.src.BatchTransitioner import TransitionFactory
from Utils.src.EnviromentHandler import VecEnvironmentHandler
from Utils.src.NodeLib.Node import Graph, PropsNode
from Utils.src.NodeLib.NodeLibrary import to_tensor, to_numpy_array, TransitionNode, BufferAppendingNode


def merge_final_observations(next_state_raw, episode_done, info):
    true_next = np.asarray(next_state_raw, dtype=np.uint8)

    if isinstance(info, dict) and "final_observation" in info:
        final_obs = info["final_observation"]
        for i, done in enumerate(episode_done):
            if done and final_obs[i] is not None:
                true_next[i] = np.asarray(final_obs[i], dtype=np.uint8)

    return true_next


class DataCollectionProcessor:
    def __init__(self, behaviour_net: nn.Module, env: VecEnvironmentHandler, buffer: ReplayBuffer,
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
            "state": np.asarray(env.reset(), dtype=np.uint8),
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
                      function=lambda s, d: torch.as_tensor(s, dtype=torch.uint8, device=d), no_grad=True),

            PropsNode("CastFloat", ["state_t"], ["state_net_input"],
                      function=lambda s: s.float(), no_grad=True),
            PropsNode("BehaviourNet", ["state_net_input"], ["q_values"], props=["behaviour_net"],
                      function=lambda net, s: net(s), no_grad=True),
            PropsNode("EpsilonGreedy", ["q_values"], ["action_raw"], props=["epsilon_greedy"],
                      function=lambda epsilon_greed, q: epsilon_greed.forward(q_values=q), no_grad=True),
            PropsNode("FormatToArray", ["action_raw"], ["action"],
                      function=to_numpy_array, no_grad=True),
            PropsNode("EnvStep", ["action"], ["next_state_raw", "reward", "terminated", "truncated", "info"],
                      props=["env"],
                      function=lambda env, a: env.step_detailed(a), no_grad=True),
            PropsNode("FormatNextState", ["next_state_raw"], ["next_state"],
                      function=lambda ns: np.asarray(ns, dtype=np.uint8), no_grad=True),
            PropsNode("CombineDones", ["terminated", "truncated"], ["done"],
                      function=lambda term, trunc: term | trunc, no_grad=True),
            PropsNode("ExtractTrueNextState", ["next_state", "done", "info"], ["real_next_state"],
                      function=merge_final_observations),
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
