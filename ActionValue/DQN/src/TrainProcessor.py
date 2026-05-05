import torch
from backend.Utils.src.NodeLib.Node import Graph, PropsNode, Signal
from torch import nn

from backend.Utils.src.NodeLib.NodeLibrary import bellman, mean_squared_error, detransition
from backend.Utils.src.NodeLib.NodeLibrary import indexing, nl_max, optimizer_normalized
from backend.Utils.src.ReplayBuffer import ReplayBuffer


class TrainProcessor:
    def __init__(self, buffer: ReplayBuffer, behavior_net: nn.Module, target_net: nn.Module,
                 optimizer: torch.optim.Optimizer, gamma: float, max_norm: float, warmup_steps: int,
                 device: torch.device):
        self.buffer = buffer
        self.behavior_net = behavior_net.to(device)
        self.target_net = target_net.to(device)
        self.optimizer = optimizer
        self.gamma = gamma
        self.device = device
        self.max_norm = max_norm
        self.warmup_steps = warmup_steps

        self.context = {
            "buffer": buffer,
            "behavior_net": behavior_net.to(device),
            "target_net": target_net.to(device),
            "optimizer": optimizer,
            "gamma": gamma,
            "max_norm": max_norm,
            "warmup_steps": warmup_steps,
            "device": device,
            "fields": buffer.spec.fields
        }

        nodes = [
            PropsNode("Sample", ["buffer", "warmup_steps"], ["batch"],
                      function=lambda b, w: b.sample_batch() if len(b) >= w else Signal.NOSIGNAL),
            PropsNode("Detransition", ["fields", "batch", "device"],
                      ["state", "action", "reward", "next_state", "done"],
                      function=detransition),
            PropsNode("BehaviorForward", ["behavior_net", "state"], ["qs_b"],
                      function=lambda net, s: net(s)),
            PropsNode("QsaBehavior", ["qs_b", "action"], ["qsa_b"],
                      function=lambda q, a: indexing(q, a.long().unsqueeze(1)).reshape(-1)),
            PropsNode("TargetForward", ["target_net", "next_state"], ["qs_t"],
                      function=lambda net, ns: net(ns), no_grad=True),
            PropsNode("Max", ["qs_t"], ["max_q_t"],
                      function=lambda qt: nl_max(qt).reshape(-1), no_grad=True),
            PropsNode("Bellman", ["max_q_t", "reward", "done", "gamma"], ["target_val"],
                      function=bellman),
            PropsNode("Loss", ["qsa_b", "target_val"], ["loss"],
                      function=mean_squared_error),
            PropsNode("Optimize", ["behavior_net", "optimizer", "loss", "max_norm"], ["_opt"],
                      function=optimizer_normalized),

            # Logging
            PropsNode("Metrics", ["loss", "qsa_b"], ["train_metrics"],
                      function=lambda l, q: {
                          "losses/td_loss": l.item(),
                          "losses/q_values": q.mean().item()
                      })
        ]

        self.graph = Graph(nodes, initial_keys=list(self.context.keys()))

    def run(self):
        self.graph.run(self.context)
        return self.context.get("train_metrics", {})
