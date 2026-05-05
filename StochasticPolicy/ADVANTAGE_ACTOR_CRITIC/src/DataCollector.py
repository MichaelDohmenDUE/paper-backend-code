import numpy as np
import torch
from torch import nn

from StochasticPolicy.PPO.discrete.src.DataCollectorGraphMujoco import merge_final_observations
from Utils.src.NodeLib.Node import Node, PropsNode, Graph
from Utils.src.RolloutBuffer import KStepRolloutBuffer
from backend.Utils.src.BatchTransitioner import TransitionFactory
from backend.Utils.src.EnviromentHandler import EnvironmentHandler
from backend.Utils.src.NodeLib.NodeLibrary import categorical_distribution, sample_distribution, \
    to_tensor, to_numpy_array, TransitionNode, BufferAppendingNode


class DataCollectionProcessor:
    def __init__(self, env_handler: EnvironmentHandler, transition_factory: TransitionFactory,
                 rollout_buffer: KStepRolloutBuffer, behaviour: nn.Module, device: torch.device):
        self.env_handler = env_handler
        self.buffer = rollout_buffer
        self.device = device
        self.t_max = 20
        self.current_t = 0

        # Logging metrics
        self.episode_timesteps = 0
        self.total_steps = 0
        self.episode_reward = 0

        self.context = {
            "state": self.env_handler.reset(),
            "behaviour": behaviour,
            "env_handler": env_handler,
            "transition_factory": transition_factory,
            "buffer": rollout_buffer,
            "device": device,
            "num_envs": getattr(self.env_handler, 'num_envs', 1),
        }

        nodes = [
            Node("FormatToTensor", ["state", "device"], ["state_t"],
                 function=lambda s, d: to_tensor(s, d).unsqueeze(0), no_grad=True),

            Node("PolicyNet", ["behaviour", "state_t"], ["logits", "value"],
                 function=lambda net, s: net(s), no_grad=False),

            Node("CreateDist", ["logits"], ["dist"],
                 function=categorical_distribution, no_grad=False),

            Node("SampleAction", ["dist"], ["action_raw", "log_prob"],
                 function=sample_distribution, no_grad=False),

            Node("FormatToArray", ["action_raw"], ["action"],
                 function=to_numpy_array, no_grad=True),

            Node("EnvStep", ["env_handler", "action"], ["next_state_raw", "reward", "done", "info"],
                 function=lambda env, a: env.step(a), no_grad=True),

            PropsNode("ExtractTrueNextState", ["next_state_raw", "done", "info"], ["true_next_state"],
                      function=merge_final_observations, no_grad=True),

            TransitionNode(
                factory=transition_factory,
                input_mapping={
                    "state": "state",
                    "logp": "log_prob",
                    "reward": "reward",
                    "done": "done",
                    "next_state": "true_next_state"
                }
            ),

            BufferAppendingNode(),

            Node("CheckTrainSignal", ["done"], ["time_to_train"],
                 function=self._update_train_signal, no_grad=True),

            Node("SetBufferState", ["buffer", "time_to_train"], ["_buffer_flag_set"],
                 function=lambda b, train_flag: b.set_ready(train_flag), no_grad=True),

            Node("StateUpdate", ["next_state_raw"], ["state"],
                 function=lambda nsr: np.array(nsr).astype(np.float32), no_grad=True)
        ]

        self.graph = Graph(nodes, initial_keys=list(self.context.keys()))

    def _update_train_signal(self, done):
        self.current_t += 1
        done_val = np.any(done) if isinstance(done, (np.ndarray, list)) else done
        if self.current_t >= self.t_max or done_val:
            self.current_t = 0
            return True
        return False

    def run(self):
        self.graph.run(self.context)

        # Logging
        reward = self.context["reward"]
        done = self.context["done"]
        time_to_train = self.context["time_to_train"]

        reward_val = np.sum(reward) if isinstance(reward, (np.ndarray, list)) else reward
        done_val = np.any(done) if isinstance(done, (np.ndarray, list)) else done

        self.episode_reward += reward_val
        self.episode_timesteps += 1
        self.total_steps += self.context["num_envs"]

        metrics = {}
        if done_val:
            metrics = {
                "charts/episodic_return": self.episode_reward,
                "charts/episodic_length": self.episode_timesteps,
                "global_step": self.total_steps,
            }
            self.episode_reward = 0
            self.episode_timesteps = 0

        return {
            "time_to_train": time_to_train,
            "metrics": metrics
        }
