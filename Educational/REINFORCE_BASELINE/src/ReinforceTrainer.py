import torch

from Utils.src.NodeLib.Node import Node, PropsNode, Graph, Signal
from backend.CommonModels.src.Policy_Reinforce_Baseline import PolicyReinforceBaseline
from backend.Utils.src import  RolloutBuffer
from backend.Utils.src.NodeLib.NodeLibrary import detransition, optimizer_update, policy_loss, mean_squared_error, \
    combined_loss
from backend.Utils.src.utils import discounted_cumulative_reward


class REINFORCETrainer:
    def __init__(self,
                 rollout_buffer: RolloutBuffer.RolloutBuffer,
                 behaviour: PolicyReinforceBaseline,
                 optimizer,
                 beta: float = 0.01,
                 gamma: float = 0.99,
                 device: torch.device = torch.device("cpu")
                 ):
        self.rollout_buffer = rollout_buffer
        self.behaviour = behaviour
        self.optimizer = optimizer
        self.beta = beta
        self.gamma = gamma
        self.device = device
        self.c_pol = 1.0
        self.c_val = 1.0

        self.context = {
            "buffer": rollout_buffer,
            "behaviour": behaviour,
            "spec_fields": rollout_buffer.spec.fields,
            "optimizer": optimizer,
            "gamma": gamma,
            "device": device,
            "c_pol": self.c_pol,
            "c_val": self.c_val
        }

        nodes = [
            PropsNode("Sample", ["buffer"], ["rollout"],
                      function=lambda b: b.sample() if b.reached_rollout_size() else Signal.NOSIGNAL),

            Node("Detransition", ["spec_fields", "rollout", "device"], ["state", "logps", "rewards", "dones"],
                 function=detransition, no_grad=False),

            Node("ComputeReturns", ["rewards", "dones", "gamma", "device"], ["G"],
                 function=lambda r, d, g, dev: torch.as_tensor(
                     discounted_cumulative_reward(g, r.cpu().numpy(), d.cpu().numpy()),
                     dtype=torch.float32, device=dev).view(-1),
                 no_grad=True),

            Node("ValueForward", ["behaviour", "state"], ["logits", "value"],
                 function=lambda net, s: net(s), no_grad=False),

            Node("FlattenValue", ["value"], ["value_flat"],
                 function=lambda v: v.view(-1), no_grad=False),

            Node("ComputeAdvantage", ["G", "value_flat"], ["advantage"],
                 function=lambda g, v: g - v.detach(), no_grad=True),

            Node("PolicyLoss", ["logps", "advantage"], ["loss_policy"],
                 function=policy_loss, no_grad=False),

            Node("ValueLoss", ["G", "value_flat"], ["loss_value"],
                 function=mean_squared_error, no_grad=False),

            Node("TotalLoss", ["loss_policy", "c_pol", "loss_value", "c_val"], ["loss"],
                 function=combined_loss, no_grad=False),

            Node("TrainStep", ["optimizer", "loss"], ["metrics"],
                 function=optimizer_update, no_grad=False)
        ]

        self.graph = Graph(nodes, initial_keys=list(self.context.keys()))

    def run(self):
        self.graph.run(self.context)

        metrics_node_output = self.context.get("metrics")
        if metrics_node_output is None or isinstance(metrics_node_output, Signal):
            return {}

        advantage = self.context["advantage"]
        loss_policy = self.context["loss_policy"]
        loss_value = self.context["loss_value"]
        loss = self.context["loss"]

        final_metrics = {
            "charts/var": advantage.var().item(),
            "losses/policy": loss_policy.item(),
            "losses/value": loss_value.item(),
            "losses/total": loss.item()
        }

        for key in ["rollout", "state", "logps", "rewards", "dones", "G",
                    "_logits", "value", "advantage", "loss_policy", "loss_value", "loss", "metrics"]:
            self.context.pop(key, None)

        return final_metrics
