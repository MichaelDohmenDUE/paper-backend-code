import torch

from Utils.src.NodeLib.Node import PropsNode, Signal, Node, Graph
from Utils.src import RolloutBuffer
from Utils.src.NodeLib.NodeLibrary import detransition, optimizer_update, policy_loss
from Utils.src.utils import discounted_cumulative_reward


class REINFORCETrainer:
    def __init__(self,
                 rollout_buffer: RolloutBuffer.RolloutBuffer,
                 optimizer,
                 beta: float = 0.01,
                 gamma: float = 0.99,
                 device: torch.device = torch.device("cpu")
                 ):
        self.rollout_buffer = rollout_buffer
        self.optimizer = optimizer
        self.device = device

        self.context = {
            "buffer": rollout_buffer,
            "spec_fields": rollout_buffer.spec.fields,
            "optimizer": optimizer,
            "gamma": gamma,
            "device": device,
        }

        nodes = [
            PropsNode("Sample", ["buffer"], ["rollout"],
                      function=lambda b: b.sample() if b.reached_rollout_size() else Signal.NOSIGNAL),

            Node("Detransition", ["spec_fields", "rollout", "device"], ["logps", "rewards", "dones"],
                 function=detransition, no_grad=False),

            Node("ComputeReturns", ["rewards", "dones", "gamma", "device"], ["G"],
                 function=lambda r, d, g, dev: torch.as_tensor(
                     discounted_cumulative_reward(g, r.cpu().numpy(), d.cpu().numpy()),
                     dtype=torch.float32, device=dev).view(-1), no_grad=True),

            Node("LossCalculation", ["logps", "G"], ["loss"],
                 function=policy_loss, no_grad=False),

            Node("TrainStep", ["optimizer", "loss"], ["metrics"],
                 function=optimizer_update, no_grad=False)
        ]

        self.graph = Graph(nodes, initial_keys=list(self.context.keys()))

    def run(self):
        self.graph.run(self.context)

        metrics_node_output = self.context.get("metrics")

        if metrics_node_output is None or isinstance(metrics_node_output, Signal):
            return {}

        G = self.context["G"]
        loss = self.context["loss"]

        final_metrics = metrics_node_output

        final_metrics.update({
            "charts/var": G.var().item(),
            "losses/policy_loss": loss.item()
        })

        for key in ["rollout", "logps", "rewards", "dones", "G", "loss", "metrics"]:
            self.context.pop(key, None)

        return final_metrics