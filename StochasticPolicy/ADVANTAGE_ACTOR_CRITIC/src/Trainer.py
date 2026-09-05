import torch

from StochasticPolicy.REINFORCE_BASELINE.src.Policy_Reinforce_Baseline import PolicyReinforceBaseline
from Utils.src.NodeLib.Node import PropsNode, Graph, Signal
from Utils.src.NodeLib.NodeLibrary import detransition, optimizer_update, policy_loss, mean_squared_error, \
    combined_loss, extract_final_states, compute_n_step_returns
from Utils.src.RolloutBuffer import KStepRolloutBuffer


class Trainer:
    def __init__(self,
                 rollout_buffer: KStepRolloutBuffer,
                 behaviour: PolicyReinforceBaseline,
                 optimizer,
                 beta: float = 0.01,
                 gamma: float = 0.99,
                 device: torch.device = torch.device("cpu"),
                 num_envs: int = 1
                 ):
        self.rollout_buffer = rollout_buffer
        self.behaviour = behaviour
        self.optimizer = optimizer
        self.beta = beta
        self.gamma = gamma
        self.device = device
        self.num_envs = num_envs
        self.c_pol = 1.0
        self.c_val = 0.5

        self.context = {
            "buffer": rollout_buffer,
            "behaviour": behaviour,
            "spec_fields": rollout_buffer.spec.fields,
            "optimizer": optimizer,
            "gamma": gamma,
            "device": device,
            "num_envs": num_envs,
            "c_pol": self.c_pol,
            "c_val": self.c_val
        }

        nodes = [
            PropsNode("Sample", ["buffer"], ["rollout"],
                      function=lambda b: b.sample() if (b.reached_rollout_size() or b.is_ready()) else Signal.NOSIGNAL),

            PropsNode("Detransition", ["spec_fields", "rollout", "device"],
                      ["state", "logps", "rewards", "dones", "next_state"],
                      function=detransition, no_grad=False),

            PropsNode("AgentForward", ["state"], ["_logits", "value"], props=["behaviour"],
                      function=lambda net, s: net(s), no_grad=False),

            PropsNode("FlattenValue", ["value"], ["value_flat"],
                      function=lambda v: v.view(-1), no_grad=False),

            PropsNode("ExtractFinalState", ["next_state", "num_envs"], ["final_next_states"],
                      function=extract_final_states, no_grad=True),

            PropsNode("BootstrapForward", ["final_next_states"], ["_nl", "next_value_final"], props=["behaviour"],
                      function=lambda net, s: net(s), no_grad=True),

            PropsNode("FlattenBootstrap", ["next_value_final"], ["next_value_flat"],
                      function=lambda v: v.view(-1), no_grad=True),

            PropsNode("ComputeReturns", ["rewards", "dones", "next_value_flat", "gamma"], ["G"],
                      function=compute_n_step_returns, no_grad=True),

            PropsNode("ComputeAdvantage", ["G", "value_flat"], ["advantage"],
                      function=lambda g, v: g - v.detach(), no_grad=True),

            PropsNode("PolicyLoss", ["logps", "advantage"], ["loss_policy"],
                      function=policy_loss, no_grad=False),

            PropsNode("ValueLoss", ["value_flat", "G"], ["loss_value"],
                      function=mean_squared_error, no_grad=False),

            PropsNode("TotalLoss", ["loss_policy", "c_pol", "loss_value", "c_val"], ["loss"],
                      function=combined_loss, no_grad=False),
            PropsNode("TrainStep", ["optimizer", "loss"], ["grad_metrics"],
                      function=optimizer_update, no_grad=False)
        ]

        self.graph = Graph(nodes, initial_keys=list(self.context.keys()))

    def run(self):
        self.graph.run(self.context)

        # Logging'
        # Check if there is anything to Extract
        grad_metrics = self.context.get("grad_metrics")
        if grad_metrics is None or isinstance(grad_metrics, Signal):
            return {}

        # Extract Metrics
        advantage = self.context["advantage"]
        loss_policy = self.context["loss_policy"]
        loss_value = self.context["loss_value"]
        loss = self.context["loss"]

        metrics = {
            "losses/policy_loss": loss_policy.item(),
            "losses/loss_value": loss_value.item(),
            "losses/total_loss": loss.item(),
            "charts/var": advantage.var().item()
        }
        return {**metrics, **grad_metrics}
