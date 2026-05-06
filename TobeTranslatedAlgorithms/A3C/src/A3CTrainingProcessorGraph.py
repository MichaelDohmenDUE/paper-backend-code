import torch

from TobeTranslatedAlgorithms.A3C.src.A3CNodes import build_a3c_graph
from backend.Utils.src.BatchTransitioner import TransitionSpec
from backend.Utils.src.NodeLib.Node import Graph

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class A3CTrainingProcessor:
    def __init__(self, global_net, local_net, optimizer, gamma, entropy_coef=0.001, max_grad_norm=40.0):
        self.global_net = global_net
        self.local_net = local_net
        self.optimizer = optimizer

        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        self.transition_spec = TransitionSpec(["state", "action", "reward", "value", "log_prob", "done", "entropy"])

        self.graph = Graph(build_a3c_graph())

    def run(self, rollout):
        if len(rollout) == 0:
            return None

        last_tr = rollout[-1]

        ctx = {
            "rollout": rollout,
            "transition_spec": self.transition_spec,

            "global_net": self.global_net,
            "local_net": self.local_net,
            "optimizer": self.optimizer,

            "gamma": self.gamma,
            "entropy_coef": self.entropy_coef,
            "max_grad_norm": self.max_grad_norm,

            "last_done": last_tr.done,
            "last_value": last_tr.value,

            "device": next(self.global_net.parameters()).device,
        }

        # Run the graph
        self.graph.run(ctx)

        return ctx.get("loss", None)
