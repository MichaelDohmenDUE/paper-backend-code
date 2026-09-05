import numpy as np
import torch

from Utils.src import ReplayBuffer, GlobalCounter
from Utils.src.BatchTransitioner import TransitionFactory
from Utils.src.EnviromentHandler import EnvironmentHandler
from Utils.src.NodeLib.Node import Graph, PropsNode
from Utils.src.NodeLib.NodeLibrary import TransitionNode, BufferAppendingNode, to_numpy_array
from Utils.src.NodeLib.NodeLibrary import noise_handler, action_with_noise


class DataCollectionProcessor:
    def __init__(self, env: EnvironmentHandler, actor, noise_generator, buffer: ReplayBuffer,
                 transition_factory: TransitionFactory, global_counter: GlobalCounter, max_action,
                 device: torch.device):
        self.context = {
            "state": np.array(env.reset()).astype(np.float32),
            "env": env,
            "actor": actor,
            "noise_generator": noise_generator,
            "buffer": buffer,
            "global_counter": global_counter,
            "max_action": max_action,
            "device": device,
            "num_envs": 1,
            "episode_reward": 0.0,
            "episode_length": 0,
            "metrics": {}
        }

        nodes = [
            PropsNode("ToTensor", ["state", "device"], ["state_tensor"],
                      function=lambda s, d: torch.as_tensor(s, dtype=torch.float32, device=d), no_grad=True),

            PropsNode("ActorForward", ["state_tensor"], ["action_tensor"], props=["actor"],
                      function=lambda net, s: net(s), no_grad=True),

            PropsNode("AddNoise", ["action_tensor", "max_action"], ["noisy_action_tensor"], props=["noise_generator"],
                      function=lambda ng, a, m: action_with_noise(ng, action_tensor=a, max_action=m), no_grad=True),
            PropsNode("ToNumpy", ["noisy_action_tensor"], ["action_np"],
                      function=to_numpy_array, no_grad=True),

            PropsNode("EnvStep", ["action_np"], ["next_state", "reward", "terminated", "truncated", "info"],
                      props=["env"],
                      function=lambda env, a: env.step_detailed(a), no_grad=True),

            PropsNode("CombineDones", ["terminated", "truncated"], ["done_reset"],
                      function=lambda term, trunc: bool(term) or bool(trunc), no_grad=True),
            PropsNode("ResetNoise", ["done_reset"], ["_dummy_noise"], props=["noise_generator"],
                      function=lambda ng, d: noise_handler(ng, done=d), no_grad=True),

            PropsNode("ExtractTrueNextState", ["next_state", "truncated", "info"], ["real_next_state"],
                      function=lambda ns, done, info: info.get("final_observation", ns) if done else ns, no_grad=True),

            TransitionNode(
                factory=transition_factory,
                input_mapping={
                    "state": "state",
                    "action": "action_np",
                    "reward": "reward",
                    "next_state": "real_next_state",
                    "done": "terminated"
                }
            ),
            BufferAppendingNode(),

            PropsNode("IncrementCounter", ["global_counter"], ["_dummy_count"],
                      function=lambda gc: gc.set(gc.get() + 1), no_grad=True),

            PropsNode("TrackMetrics", ["reward", "done_reset"], ["_dummy_track"],
                      function=self._track_helper, no_grad=True),

            PropsNode("StateUpdate", ["next_state", "_buffer_updated"], ["state"],
                      function=lambda ns, signal: np.array(ns).astype(np.float32)),
        ]

        self.graph = Graph(nodes, initial_keys=list(self.context.keys()))

    def _track_helper(self, reward, done_reset):
        self.context["episode_reward"] += float(reward[0])
        self.context["episode_length"] += 1

        if done_reset:
            self.context["metrics"] = {
                "charts/episodic_return": self.context["episode_reward"],
                "charts/episodic_length": self.context["episode_length"],
                "global_step": self.context["global_counter"].get()
            }
            self.context["episode_reward"] = 0.0
            self.context["episode_length"] = 0

    def run(self):
        self.context["metrics"] = {}
        self.graph.run(self.context)
        return self.context["metrics"]
