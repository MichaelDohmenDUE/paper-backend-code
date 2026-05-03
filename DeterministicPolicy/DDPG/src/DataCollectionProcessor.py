import numpy as np
import torch
from backend.CommonModels.src.Policy import Policy
from backend.Utils.src import ReplayBuffer, GlobalCounter
from backend.Utils.src.BatchTransitioner import TransitionFactory
from backend.Utils.src.EnviromentHandler import EnvironmentHandler
from backend.Utils.src.NodeLib.NodeLibrary import reset_handler, noise_handler, action_with_noise

from NodeLib.Node import Graph, PropsNode
from NodeLib.NodeLibrary import TransitionNode, BufferAppendingNode, to_numpy_array


class DataCollectionProcessor:
    def __init__(self, env: EnvironmentHandler, actor, noise_generator, buffer: ReplayBuffer,
                 transition_factory: TransitionFactory, global_counter: GlobalCounter, max_action,
                 device: torch.device):
        self.num_envs = getattr(env, "num_envs", 1)

        self.context = {
            "state": np.array(env.reset()).astype(np.float32),
            "env": env,
            "actor": actor,
            "noise_generator": noise_generator,
            "buffer": buffer,
            "global_counter": global_counter,
            "max_action": max_action,
            "device": device,
            "num_envs": self.num_envs
        }

        nodes = [
            PropsNode("ToTensor", ["state", "device"], ["state_tensor"],
                      function=lambda s, d: torch.as_tensor(s, dtype=torch.float32, device=d), no_grad=True),

            PropsNode("ActorForward", ["actor", "state_tensor"], ["action_tensor"],
                      function=lambda net, s: net(s), no_grad=True),

            PropsNode("AddNoise", ["noise_generator", "action_tensor", "max_action"], ["noisy_action_tensor"],
                      function=lambda ng, a, m: action_with_noise(ng, action_tensor=a, max_action=m), no_grad=True),
            PropsNode("ToNumpy", ["noisy_action_tensor"], ["action_np"],
                      function=to_numpy_array, no_grad=True),

            PropsNode("EnvStep", ["env", "action_np"], ["next_state", "reward", "terminated", "truncated", "info"],
                      function=lambda env, a: env.step_ddpg(a), no_grad=True),

            PropsNode("CombineDones", ["terminated", "truncated"], ["done_reset"],
                      function=lambda terminated, truncated: terminated | truncated, no_grad=True),
            PropsNode("ResetNoise", ["noise_generator", "done_reset"], ["_dummy_noise"],
                      function=lambda ng, d: noise_handler(ng, done=np.any(d)), no_grad=True),
            PropsNode("ExtractTrueNextState", ["next_state", "terminated", "truncated", "info"], ["real_next_state"],
                      function=lambda ns, terminated, truncated, info: info.get("final_observation", ns) if (
                              np.any(terminated) or np.any(truncated)) else ns, no_grad=True),

            # 5. Buffer Transition
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

            # 6. Metrics and State Overwrite
            PropsNode("IncrementCounter", ["global_counter"], ["_dummy_count"],
                      function=lambda gc: gc.set(gc.get() + 1), no_grad=True),

            PropsNode("StateUpdate", ["next_state", "_buffer_updated"], ["state"],
                      function=lambda ns, signal: np.array(ns).astype(np.float32)),
        ]

        self.graph = Graph(nodes, initial_keys=list(self.context.keys()))

    def run(self):
        self.graph.run(self.context)
        metrics = {}
        return metrics
