import torch
from torch import nn

from backend.Utils.src.NodeLib.Node import Signal, Graph, PropsNode
from backend.Utils.src.NodeLib.NodeLibrary import bellman, detransition, indexing, mean_squared_error
from backend.Utils.src.NodeLib.NodeLibrary import optimizer_update
from backend.Utils.src.ReplayBuffer import ReplayBuffer


class TrainProcessor:
    def __init__(self, buffer: ReplayBuffer, behavior_net: nn.Module, target_net: nn.Module,
                 optimizer: torch.optim.Optimizer, gamma: float, device: torch.device, warmup_steps=500):
        self.buffer = buffer
        self.behavior_net = behavior_net.to(device)
        self.target_net = target_net.to(device)
        self.optimizer = optimizer
        self.gamma = gamma
        self.warmup_steps = warmup_steps
        self.device = device

        self.context = {
            "buffer": buffer,
            "behavior_net": behavior_net.to(device),
            "target_net": target_net.to(device),
            "optimizer": optimizer,
            "gamma": gamma,
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

            PropsNode("BehaviorForward", ["state"], ["qs_b"], props=["behavior_net"],
                      function=lambda net, s: net(s.float())),
            PropsNode("QsaBehavior", ["qs_b", "action"], ["qsa_b"],
                      function=lambda q, a: indexing(q, a.long().unsqueeze(1)).reshape(-1)),
            PropsNode("BehaviorNextForward", ["next_state"], ["next_qs_b"], props=["behavior_net"],
                      function=lambda net, ns: net(ns.float()), no_grad=True),
            PropsNode("SelectNextAction", ["next_qs_b"], ["next_actions"],
                      function=lambda q: torch.argmax(q, dim=1).view(-1, 1), no_grad=True),
            PropsNode("TargetForward", ["next_state"], ["qs_t"], props=["target_net"],
                      function=lambda net, ns: net(ns.float()), no_grad=True),
            PropsNode("QsaTarget", ["qs_t", "next_actions"], ["qsa_t"],
                      function=lambda qt, a: indexing(qt, a).reshape(-1), no_grad=True),
            PropsNode("Bellman", ["qsa_t", "reward", "done", "gamma"], ["target_val"],
                      function=bellman),
            PropsNode("Loss", ["qsa_b", "target_val"], ["loss"],
                      function=mean_squared_error),
            PropsNode("Optimize", ["loss"], ["_opt"], props=["optimizer"],
                      function=optimizer_update),

            PropsNode("Metrics", ["loss", "qsa_b"], ["train_metrics"],
                      function=lambda l, q: {
                          "losses/td_loss": l.item(),
                          "losses/q_values": q.mean().item()
                      })
        ]

        self.graph = Graph(nodes, initial_keys=list(self.context.keys()))

    def run(self):
        self.graph.run(self.context)
        res = self.context.get("train_metrics", {})
        if isinstance(res, Signal) or res is Signal.NOSIGNAL:
            return {}
        return res
