import numpy as np

from Utils.src.NodeLib.Node import Graph, PropsNode
from Utils.src.NodeLib.NodeLibrary import categorical_distribution, sample_distribution, \
    to_tensor, TransitionNode, BufferAppendingNode, to_numpy_array


class DataCollectionProcessor:
    def __init__(self, env_handler, transition_factory, rollout_buffer, behaviour, device):
        self.env_handler = env_handler
        self.device = device

        # Initializing metrics
        self.episode_reward = 0
        self.episode_timesteps = 0
        self.total_steps = 0

        # Initial context
        self.context = {
            "state": self.env_handler.reset(),
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

            PropsNode("PolicyNet", ["state_t"], ["logits"], props=["behaviour"],
                      function=lambda net, s: net(s), no_grad=False),

            PropsNode("CreateDist", ["logits"], ["dist"],
                      function=categorical_distribution, no_grad=False),

            PropsNode("SampleAction", ["dist"], ["action_raw", "log_prob"],
                      function=sample_distribution, no_grad=False),

            PropsNode("FormatToArray", ["action_raw"], ["action"],
                      function=to_numpy_array, no_grad=True),

            PropsNode("EnvStep", ["action"], ["next_state", "reward", "done", "info"], props=["env_handler"],
                      function=lambda env, a: env.step(a), no_grad=True),

            TransitionNode(
                factory=transition_factory,
                input_mapping={
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
        """Executes one step in the environment."""
        self.graph.run(self.context)

        # Extract values for logging
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
