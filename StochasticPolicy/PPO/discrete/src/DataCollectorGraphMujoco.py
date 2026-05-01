import numpy as np
import torch

from NodeLib.NodeLibrary import to_tensor, to_numpy_array
from backend.StochasticPolicy.PPO.discrete.src.DataCollectorGraph import EpisodicMetricsNode
from backend.Utils.src.NodeLib.Node import Node, Graph
from backend.Utils.src.NodeLib.NodeLibrary import TransitionNode, BufferAppendingNode, BootStrappingNode, \
    BootStrappingNodeMujoco
from backend.Utils.src.RolloutBuffer import RolloutBuffer

class DataCollectionProcessor:
    def __init__(self, env_handler, transition_factory, rollout_buffer: RolloutBuffer, rollout_size, actor, critic,
                 device):
        self.num_envs = getattr(env_handler, "num_envs", 1)

        self.context = {
            "state": np.array(env_handler.reset()).astype(np.float32),
            "buffer": rollout_buffer,
            "env": env_handler,
            "actor": actor,
            "critic": critic,
            "device": device,
            "running_rewards": np.zeros(self.num_envs),
            "metrics_queue": [],
            "total_steps": 0,
            "num_envs": self.num_envs
        }

        nodes = [
            Node("ToTensor", ["state", "device"], ["state_tensor"], function=to_tensor, no_grad=True),
            Node("ActorForward", ["actor", "state_tensor"], ["dist"], function=lambda net, s: net(s)),
            Node("CriticForward", ["critic", "state_tensor"], ["value_tensor"], function=lambda net, s: net(s)),
            Node("SampleAction", ["dist"], ["action_tensor"], function=lambda dist: dist.sample(), no_grad=True),
            Node("LogProb", ["dist", "action_tensor"], ["logp_tensor"], function=lambda dist, a: dist.log_prob(a), no_grad=True),
            Node("SqueezeValue", ["value_tensor"], ["value_t_sq"], function=lambda v: v.squeeze(-1), no_grad=True),# TODO: Rewirte
            Node("ToNumpy_a", ["action_tensor"], ["action"], function=to_numpy_array, no_grad=True),
            Node("ToNumpy_l", ["logp_tensor"], ["logp"], function=to_numpy_array, no_grad=True),
            Node("ToNumpy_v", ["value_t_sq"], ["value"], function=to_numpy_array, no_grad=True),
            Node("EnvStep", ["env", "action"], ["next_state", "reward", "done", "info"],
                 function=lambda env, a: env.step(a)),

            TransitionNode(
                factory=transition_factory,
                input_mapping={
                    "state": "state",
                    "action": "action",
                    "logp": "logp",
                    "reward": "reward",
                    "done": "done",
                    "value": "value"
                },
                default_kwargs={
                    "bootstrap_value": 0.0
                }
            ),
            BufferAppendingNode(),

            BootStrappingNodeMujoco(rollout_size),

            Node("State", ["next_state", "_buffer_updated"], ["state"],
                 function=lambda ns, _: np.array(ns).astype(np.float32)),
            EpisodicMetricsNode(),
            Node("CountSteps", ["total_steps", "num_envs"], ["total_steps"], function=lambda steps, n: steps + n),

        ]

        self.graph = Graph(nodes, initial_keys=list(self.context.keys()))

    def run(self):
        self.graph.run(self.context)

        combined_metrics = {}

        if self.context["buffer"].reached_rollout_size() and self.context["metrics_queue"]:
            returns = [m["charts/episodic_return"] for m in self.context["metrics_queue"]]
            combined_metrics["charts/episodic_return"] = np.mean(returns)
            self.context["metrics_queue"].clear()

        return combined_metrics
