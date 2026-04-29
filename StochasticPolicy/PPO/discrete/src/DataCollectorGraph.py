import numpy as np
import torch

from backend.Utils.src.NodeLib.NodeLibrary import TransitionNode, BufferAppendingNode, BootStrappingNode
from backend.Utils.src.RolloutBuffer import RolloutBuffer
from backend.Utils.src.NodeLib.Node import Node, Graph


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
            "num_envs": self.num_envs
        }

        nodes = [
            Node("ToTensor", ["state", "device"], ["state_t"],
                 function=lambda s, d: torch.as_tensor(s, dtype=torch.float32, device=d), no_grad=True),
            Node("AgentForward", ["agent", "state_t"], ["dist", "value_t"], function=lambda net, s: net(s), no_grad=True),
            Node("SampleAction", ["dist"], ["action_t"], function=lambda dist: dist.sample(), no_grad=True),
            Node("LogProb", ["dist", "action_t"], ["logp_t"], function=lambda dist, a: dist.log_prob(a), no_grad=True),
            Node("SqueezeValue", ["value_t"], ["value_t_sq"], function=lambda v: v.squeeze(-1), no_grad=True),
            Node("ToNumpy_a", ["action_t"], ["action"], function=lambda t: t.cpu().numpy().squeeze(), no_grad=True),
            Node("ToNumpy_l", ["logp_t"], ["logp"], function=lambda t: t.cpu().numpy().squeeze(), no_grad=True),
            Node("ToNumpy_v", ["value_t_sq"], ["value"], function=lambda t: t.cpu().numpy().squeeze(), no_grad=True),
            Node("EnvStep", ["env", "action"], ["next_state", "reward", "done", "info"],
                 function=lambda env, a: env.step(a)),
            Node("ClipReward", ["reward"], ["clipped_reward"], function=lambda r: np.clip(r, -1, 1)),

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

            EpisodicMetricsNode(),
            Node("CountSteps", ["total_steps", "num_envs"], ["total_steps"], function=lambda steps, n: steps + n),

            BootStrappingNode(rollout_size),
            Node("State", ["next_state", "_buffer_updated"], ["state"],
                 function=lambda ns, _: np.array(ns).astype(np.uint8))
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