import numpy as np
from backend.Utils.src.NodeLib.Node import Node, Graph, PropsNode
from backend.Utils.src.NodeLib.NodeLibrary import TransitionNode, BufferAppendingNode, to_numpy_array, to_tensor
from backend.Utils.src.RolloutBuffer import RolloutBuffer

def merge_final_observations(next_state_raw, episode_done, info):
    true_next = np.array(next_state_raw).copy()
    if isinstance(info, dict) and "final_observation" in info:
        final_obs = info["final_observation"]
        for i, done in enumerate(episode_done):
            if done and final_obs[i] is not None:
                true_next[i] = final_obs[i]
    return true_next.astype(np.uint8)

class EpisodicMetricsNode(Node):
    def __init__(self):
        super().__init__("EpisodicMetrics", ["running_rewards", "metrics_queue", "reward", "episode_done"], [])

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
    def __init__(self, env_handler, transition_factory, rollout_buffer: RolloutBuffer, rollout_size, agent, device):
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
            PropsNode("AgentForward", ["agent", "state_tensor"], ["dist", "value_tensor"],
                      function=lambda net, s: net(s.float()), no_grad=True),
            PropsNode("SampleAction", ["dist"], ["action_tensor"], function=lambda dist: dist.sample(), no_grad=True),
            PropsNode("LogProb", ["dist", "action_tensor"], ["logp_tensor"], function=lambda dist, a: dist.log_prob(a),
                      no_grad=True),
            PropsNode("SqueezeValue", ["value_tensor"], ["value_t_sq"], function=lambda v: v.squeeze(-1), no_grad=True),
            PropsNode("ToNumpy_a", ["action_tensor"], ["action"], function=to_numpy_array, no_grad=True),
            PropsNode("ToNumpy_l", ["logp_tensor"], ["logp"], function=to_numpy_array, no_grad=True),
            PropsNode("ToNumpy_v", ["value_t_sq"], ["value"], function=to_numpy_array, no_grad=True),
            PropsNode("EnvStep", ["env", "action"], ["next_state_raw", "reward", "terminated", "truncated", "info"],
                      function=lambda env, a: env.step_detailed(a), no_grad=True),

            PropsNode("ExtractTrueNextState", ["next_state_raw", "truncated", "info"], ["next_state"],
                      function=merge_final_observations, no_grad=True),

            PropsNode("ClipReward", ["reward"], ["clipped_reward"],
                      function=lambda r: np.sign(r), no_grad=True),

            TransitionNode(
                factory=transition_factory,
                input_mapping={
                    "state": "state",
                    "action": "action",
                    "logp": "logp",
                    "reward": "clipped_reward",
                    "terminated": "terminated",
                    "next_state": "next_state",
                },
            ),
            BufferAppendingNode(),

            PropsNode("CombineDones", ["terminated", "truncated"], ["episode_done"],
                      function=lambda term, trunc: term | trunc, no_grad=True),

            PropsNode("State", ["next_state_raw", "_buffer_updated"], ["state"],
                      function=lambda ns, _: np.array(ns).astype(np.uint8), no_grad=True),
            EpisodicMetricsNode(),

            PropsNode("CountSteps", ["total_steps", "num_envs"], ["total_steps"], function=lambda steps, n: steps + n),
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
