import torch
import torch.nn.functional as F
from torch import nn

from backend.Utils.src.NodeLib.Node import Node, Signal, Graph
from backend.Utils.src.NodeLib.NodeLibrary import bellman, detransition, indexing, argmax, mean_squared_error, \
    optimizer_normalized
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
            "max_norm": 0.5,
            "device": device,
            "fields": buffer.spec.fields
        }

        nodes = [
            Node("Sample", ["buffer", "warmup_steps"], ["batch"],
                 function=lambda b, w: b.sample_batch() if len(b) >= w else Signal.NOSIGNAL),

            Node("Detransition", ["fields", "batch", "device"], ["state", "action", "reward", "next_state", "done"],
                 function=detransition),

            Node("BehaviorForward", ["behavior_net", "state"], ["qs_b"],
                 function=lambda net, s: net(s.float().squeeze(1))),
            Node("QsaBehavior", ["qs_b", "action"], ["qsa_b"],
                 function=lambda q, a: indexing(q, a).reshape(-1)),
            Node("BehaviorNextForward", ["behavior_net", "next_state"], ["next_qs_b"],
                 function=lambda net, ns: net(ns.float().squeeze(1)), no_grad=True),
            Node("SelectNextAction", ["next_qs_b"], ["next_actions"],
                 function=lambda q: torch.argmax(q, dim=1).view(-1, 1), no_grad=True),
            Node("TargetForward", ["target_net", "next_state"], ["qs_t"],
                 function=lambda net, ns: net(ns.float().squeeze(1)), no_grad=True),
            Node("QsaTarget", ["qs_t", "next_actions"], ["qsa_t"],
                 function=lambda qt, a: indexing(qt, a).reshape(-1), no_grad=True),
            Node("Bellman", ["qsa_t", "reward", "done", "gamma"], ["target_val"],
                 function=bellman),
            Node("Loss", ["qsa_b", "target_val"], ["loss"],
                 function=mean_squared_error),

            Node("Optimize", ["behavior_net", "optimizer", "loss", "max_norm"], ["_opt"],
                 function=optimizer_normalized),

            Node("Metrics", ["loss", "qsa_b"], ["train_metrics"],
                 function=lambda l, q: {
                     "losses/td_loss": l.item(),
                     "losses/q_values": q.mean().item()
                 })
        ]

        self.graph = Graph(nodes, initial_keys=list(self.context.keys()))

    def run(self):
        self.graph.run(self.context)
        return self.context.get("train_metrics", {})

