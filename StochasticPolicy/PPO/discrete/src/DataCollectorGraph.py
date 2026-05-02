import numpy as np
import torch

from backend.Utils.src.NodeLib.NodeLibrary import TransitionNode, BufferAppendingNode, BootStrappingNode, \
    to_numpy_array, to_tensor, clipper
from backend.Utils.src.RolloutBuffer import RolloutBuffer
from backend.Utils.src.NodeLib.Node import Node, Graph, PropsNode


class EpisodicMetricsNode(Node):
    def __init__(self):
        super().__init__("EpisodicMetrics", ["running_rewards", "metrics_queue", "reward", "done"], [])

    def forward(self, running_rewards, metrics_queue, reward, done):
        reward = np.atleast_1d(reward)
        done = np.atleast_1d(done)

        running_rewards += reward

        for i in range(len(done)):
            if done[i]:
                metrics_queue.append({
                    "charts/episodic_return": running_rewards[i]
                })
                running_rewards[i] = 0

class DataCollectionProcessor:
    def __init__(self, env_handler, transition_factory, rollout_buffer: RolloutBuffer, rollout_size, agent,device):
        self.num_envs = getattr(env_handler, "num_envs", 1)

        self.context = {
            "state": np.array(env_handler.reset()).astype(np.uint8),
            "buffer": rollout_buffer,
            "env": env_handler,
            "agent": agent,
            "device": device,
            "running_rewards": np.zeros(self.num_envs),
            "metrics_queue": [],
            "total_steps": 0,
            "num_envs": self.num_envs,
            "max_R": 1,
        }

        nodes = [
            PropsNode("ToTensor", ["state", "device"], ["state_tensor"], function=to_tensor, no_grad=True),
            PropsNode("AgentForward", ["agent", "state_tensor"], ["dist", "value_tensor"], function=lambda net, s: net(s), no_grad=True),
            PropsNode("SampleAction", ["dist"], ["action_tensor"], function=lambda dist: dist.sample(), no_grad=True),
            PropsNode("LogProb", ["dist", "action_tensor"], ["logp_tensor"], function=lambda dist, a: dist.log_prob(a), no_grad=True),
            PropsNode("SqueezeValue", ["value_tensor"], ["value_t_sq"], function=lambda v: v.squeeze(-1), no_grad=True),
            PropsNode("ToNumpy_a", ["action_tensor"], ["action"], function=to_numpy_array, no_grad=True),
            PropsNode("ToNumpy_l", ["logp_tensor"], ["logp"], function=to_numpy_array, no_grad=True),
            PropsNode("ToNumpy_v", ["value_t_sq"], ["value"], function=to_numpy_array, no_grad=True),
            PropsNode("EnvStep", ["env", "action"], ["next_state", "reward", "done", "info"],
                      function=lambda env, a: env.step(a)),
            PropsNode("ClipReward", ["reward", "max_R"], ["clipped_reward"], function=clipper),

            TransitionNode(
                factory=transition_factory,
                input_mapping={
                    "state": "state",
                    "action": "action",
                    "logp": "logp",
                    "reward": "clipped_reward",
                    "done": "done",
                    "value": "value"
                },
                default_kwargs={
                    "bootstrap_value": 0.0
                }
            ),
            BufferAppendingNode(),
            BootStrappingNode(rollout_size),

            Node("OverwriteState", ["next_state", "_buffer_updated"], ["state"],
                 function=lambda ns, _: np.array(ns).astype(np.uint8)),

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