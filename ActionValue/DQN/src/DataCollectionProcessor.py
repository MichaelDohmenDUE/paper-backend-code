import numpy as np
import torch
import wandb
from torch import nn

from backend.ActionValue.DQN.src.ActionHandler import EpsilonGreedyPolicy
from backend.Utils.src.NodeLib.Node import Node, Graph
from backend.Utils.src.ReplayBuffer import ReplayBuffer
from backend.Utils.src.BatchTransitioner import TransitionFactory
from backend.Utils.src.EnviromentHandler import VecEnvironmentHandler
from backend.Utils.src.NodeLib.NodeLibrary import reset_handler, TransitionNode, BufferAppendingNode


class DataCollectionProcessor:
    def __init__(self, behaviour: nn.Module, env: VecEnvironmentHandler, buffer: ReplayBuffer,
                 epsilon_greedy: EpsilonGreedyPolicy, transition_factory: TransitionFactory, device: torch.device):
        self.env = env
        self.epsilon_greedy = epsilon_greedy
        # Logging
        self.running_rewards = np.zeros(self.env.num_envs)
        self.running_lengths = np.zeros(self.env.num_envs)
        self.total_steps = 0

        self.context = {
            "state": np.array(env.reset()).astype(np.float32),
            "behaviour": behaviour,
            "env": env,
            "buffer": buffer,
            "epsilon_greedy": epsilon_greedy,
            "transition_factory": transition_factory,
            "device": device,
            "num_envs": self.env.num_envs,
        }

        nodes = [
            Node("ToTensor", ["state", "device"], ["state_t"],
                 function=lambda s, d: torch.as_tensor(s, dtype=torch.float32, device=d), no_grad=True),
            Node("BehaviourNet", ["behaviour", "state_t"], ["q_values"],
                 function=lambda net, s: net(s), no_grad=True),
            Node("EpsilonGreedy", ["epsilon_greedy", "q_values"], ["action_raw"],
                 function=lambda epsilon_greed, q: epsilon_greed.forward(q_values=q), no_grad=True),

            Node("FormatAction", ["action_raw"], ["action"],
                 function=lambda a: np.atleast_1d(a.cpu().numpy() if torch.is_tensor(a) else a)),

            Node("EnvStep", ["env", "action"], ["next_state", "reward", "done", "info"],
                 function=lambda env, a: env.step(a)),

            TransitionNode(
                factory=transition_factory,
                input_mapping={
                    "state": "state",
                    "action": "action",
                    "reward": "reward",
                    "next_state": "next_state",
                    "done": "done"
                }
            ),
            BufferAppendingNode(),
            Node("StateUpdate", ["next_state", "_buffer_updated"], ["state"],
                 function=lambda ns, signal: np.array(ns).astype(np.float32)),
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


