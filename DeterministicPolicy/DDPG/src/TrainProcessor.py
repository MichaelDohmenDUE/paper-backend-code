import torch
import torch.nn.functional as F
from torch import nn

from backend.Utils.src.NodeLib.NodeLibrary import bellman, optimizer_update, detransition, mean_squared_error, \
    deterministic_policy_gradient
from backend.Utils.src.ReplayBuffer import ReplayBuffer

from NodeLib.Node import PropsNode, Graph

class TrainProcess:
    def __init__(self, replay_buffer: ReplayBuffer, actor: nn.Module, actor_target: nn.Module, critic: nn.Module,
                 critic_target: nn.Module, actor_optimizer: torch.optim.Optimizer,
                 critic_optimizer: torch.optim.Optimizer, gamma: float, warmup: int, device: torch.device):
        self.replay_buffer = replay_buffer
        self.warmup = warmup
        self.actor = actor

        self.context = {
            "buffer": replay_buffer,
            "actor": actor.to(device),
            "actor_target": actor_target.to(device),
            "critic": critic.to(device),
            "critic_target": critic_target.to(device),
            "actor_opt": actor_optimizer,
            "critic_opt": critic_optimizer,
            "gamma": gamma,
            "device": device,
            "spec_fields": replay_buffer.spec.fields
        }

        nodes = [
            PropsNode("SampleBatch", ["buffer"], ["batch"],
                      function=lambda b: b.sample_batch()),
            PropsNode("Detransition", ["spec_fields", "batch", "device"],
                      ["state", "action", "reward", "next_state", "done"],
                      function=detransition),

            PropsNode("TargetAction", ["actor_target", "next_state"], ["next_action"],
                      function=lambda net, s: net(s), no_grad=True),
            PropsNode("TargetQ", ["critic_target", "next_state", "next_action"], ["target_q_val"],
                      function=lambda net, s, a: net(s, a).squeeze(), no_grad=True),
            PropsNode("TargetBellman", ["target_q_val", "reward", "done", "gamma"], ["target"],
                      function=bellman, no_grad=True),

            PropsNode("CurrentQ", ["critic", "state", "action"], ["current_q"],
                      function=lambda net, s, a: net(s, a).squeeze()),

            PropsNode("CriticLoss", ["current_q", "target"], ["critic_loss"],
                      function=mean_squared_error),

            PropsNode("UpdateCritic", ["critic_opt", "critic_loss"], ["_c_opt"],
                      function=optimizer_update),

            PropsNode("ActorForward", ["actor", "state", "_c_opt"], ["policy_action"],
                      function=lambda net, s, _: net(s)),

            PropsNode("ActorQValue", ["critic", "state", "policy_action"], ["actor_q_val"],
                      function=lambda net, s, a: net(s, a).squeeze()),

            PropsNode("ActorLoss", ["actor_q_val"], ["actor_loss"],
                      function=deterministic_policy_gradient),
            PropsNode("UpdateActor", ["actor_opt", "actor_loss"], ["_a_opt"],
                      function=optimizer_update),

            PropsNode("ExtractMetrics", ["critic_loss", "actor_loss"], ["metrics"],
                      function=lambda critic_l, actor_l: {
                          "losses/critic_loss": critic_l.item(),
                          "losses/actor_loss": actor_l.item()
                      }, no_grad=True)
        ]

        self.graph = Graph(nodes, initial_keys=list(self.context.keys()))

    def run(self):
        if len(self.replay_buffer) < self.replay_buffer.batch_size or len(self.replay_buffer) < self.warmup:
            return {}

        self.graph.run(self.context)
        return self.context.get("metrics", {})
