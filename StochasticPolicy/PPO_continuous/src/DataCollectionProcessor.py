import numpy as np

from StochasticPolicy.PPO_discrete_Mujoco.src.DataCollectorGraphMujoco import merge_final_observations
from StochasticPolicy.PPO_discrete_Atari.PPO_Atari_Baseline.src import EpisodicMetricsNode
from Utils.src.NodeLib.Node import Node, Graph
from Utils.src.NodeLib.Node import PropsNode
from Utils.src.NodeLib.NodeLibrary import TransitionNode, BufferAppendingNode
from Utils.src.NodeLib.NodeLibrary import to_tensor, to_numpy_array
from Utils.src.RolloutBuffer import RolloutBuffer


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
            PropsNode("ToTensor", ["state", "device"], ["state_tensor"], function=to_tensor, no_grad=True),
            PropsNode("ActorForward", ["state_tensor"], ["dist"], props=["actor"],
                      function=lambda net, s: net(s), no_grad=True),
            PropsNode("CriticForward", ["state_tensor"], ["value_tensor"], props=["critic"],
                      function=lambda net, s: net(s), no_grad=True),
            PropsNode("SampleAction", ["dist"], ["action_tensor"], function=lambda dist: dist.sample(), no_grad=True),
            PropsNode("LogProb", ["dist", "action_tensor"], ["logp_tensor"],
                      function=lambda dist, a: dist.log_prob(a).sum(dim=-1) if len(
                          dist.log_prob(a).shape) > 1 else dist.log_prob(a), no_grad=True),

            PropsNode("SqueezeValue", ["value_tensor"], ["value_t_sq"], function=lambda v: v.squeeze(-1), no_grad=True),

            PropsNode("ToNumpy_a", ["action_tensor"], ["action"], function=to_numpy_array, no_grad=True),
            PropsNode("ToNumpy_l", ["logp_tensor"], ["logp"], function=to_numpy_array, no_grad=True),
            PropsNode("ToNumpy_v", ["value_t_sq"], ["value"], function=to_numpy_array, no_grad=True),

            PropsNode("ClipAction", ["action"], ["clipped_action"], function=lambda a: np.clip(a, -1.0, 1.0)),

            PropsNode("EnvStep", ["action"], ["next_state_raw", "reward", "terminated", "truncated", "info"],
                      props=["env"],
                      function=lambda env, a: env.step_detailed(a)),

            PropsNode("ExtractTrueNextState", ["next_state_raw", "truncated", "info"], ["next_state"],
                      function=merge_final_observations, no_grad=True),

            TransitionNode(
                factory=transition_factory,
                input_mapping={
                    "state": "state",
                    "action": "action",
                    "logp": "logp",
                    "reward": "reward",
                    "terminated": "terminated",
                    "next_state": "next_state",
                },
            ),
            BufferAppendingNode(),

            PropsNode("CombineDones", ["terminated", "truncated"], ["episode_done"],
                      function=lambda term, trunc: term | trunc, no_grad=True),

            Node("State", ["next_state_raw", "_buffer_updated"], ["state"],
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
