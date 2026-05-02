import numpy as np
import torch

from backend.Utils.src.EnviromentHandler import VecEnvironmentHandler
from backend.Utils.src.GlobalCounter import GlobalCounter
from NodeLib.Node import PropsNode, Graph
from backend.Utils.src.BatchTransitioner import TransitionFactory
from backend.Utils.src.NodeLib.NodeLibrary import BufferAppendingNode, TransitionNode, ConditionNode
from backend.Utils.src.ReplayBuffer import ReplayBuffer


def random_action(env, behaviour, state, expl_noise, max_action, action_size, device):
    num_envs = getattr(env, "num_envs", 1)
    return np.random.uniform(-max_action, max_action, (num_envs, action_size))


def network_action(env, behaviour, state, expl_noise, max_action, action_size, device):
    state_t = torch.as_tensor(state, dtype=torch.float32, device=device)
    if state_t.dim() == 1:
        state_t = state_t.unsqueeze(0).to(device)
    action = behaviour(state_t).detach().cpu().numpy()

    num_envs = action.shape[0]
    noise = np.random.normal(0, max_action * expl_noise, size=(num_envs, action_size))
    action = np.clip(action + noise, -max_action, max_action)
    return action


class DataCollectionProcessor:
    def __init__(self, behaviour, env_handler: VecEnvironmentHandler, transition_factory: TransitionFactory,
                 replay_buffer: ReplayBuffer, global_counter: GlobalCounter,
                 max_action, expl_noise, warmup_steps, action_size, device):

        self.num_envs = getattr(env_handler, "num_envs", 1)

        self.context = {
            "state": np.array(env_handler.reset()),
            "env": env_handler,
            "behaviour": behaviour,
            "buffer": replay_buffer,
            "global_counter": global_counter,
            "max_action": max_action,
            "expl_noise": expl_noise,
            "warmup_steps": warmup_steps,
            "device": device,
            "running_rewards": np.zeros(self.num_envs),
            "running_lengths": np.zeros(self.num_envs),
            "num_envs": self.num_envs,
            "action_size": action_size
        }
        nodes = [
            PropsNode("GetStep", ["global_counter"], ["current_step"],
                      function=lambda gc: gc.get(), no_grad=True),

            PropsNode("CheckWarmup", ["current_step", "warmup_steps"], ["is_warmup"],
                      function=lambda step, warmup: step < warmup, no_grad=True),

            ConditionNode(
                name="SelectAction",
                condition_key="is_warmup",
                func_1=network_action,
                func_2=random_action,
                inputs=["env", "behaviour", "state", "expl_noise", "max_action", "action_size", "device"],
                outputs=["action"],
                no_grad=True
            ),
            PropsNode("EnvStep", ["env", "action"], ["next_state", "reward", "terminated", "truncated", "info"],
                      function=lambda env, a: env.step_detailed(a), no_grad=True),

            PropsNode("CombineDones", ["terminated", "truncated"], ["done_reset"],
                      function=lambda term, trunc: term | trunc, no_grad=True),

            TransitionNode(
                factory=transition_factory,
                input_mapping={
                    "state": "state", "action": "action", "reward": "reward",
                    "next_state": "next_state", "done": "terminated"
                }
            ),
            BufferAppendingNode(),

            PropsNode("IncrementCounter", ["global_counter"], ["_dummy"],
                      function=lambda gc: gc.set(gc.get() + 1), no_grad=True),

            PropsNode("TrackMetrics", ["reward", "done_reset", "running_rewards", "running_lengths"],
                      ["running_rewards", "running_lengths", "metrics"],
                      function=self._track_helper, no_grad=True),

            PropsNode("StateUpdate", ["next_state", "_buffer_updated"], ["state"],
                      function=lambda ns, signal: np.array(ns).astype(np.float32)),
        ]

        self.graph = Graph(nodes, initial_keys=list(self.context.keys()))

    @staticmethod
    def _track_helper(reward, done_reset, running_rewards, running_lengths):
        running_rewards += reward
        running_lengths += 1

        metrics = {}
        dones = np.atleast_1d(done_reset)

        for i in range(len(dones)):
            if dones[i]:
                metrics = {
                    "charts/episodic_return": float(running_rewards[i]),
                    "charts/episodic_length": int(running_lengths[i])
                }
                running_rewards[i] = 0
                running_lengths[i] = 0
        return running_rewards, running_lengths, metrics

    def run(self):
        self.graph.run(self.context)
        metrics = self.context.get("metrics", {})
        if metrics:
            metrics["global_step"] = self.context["global_counter"].get()
        return metrics
