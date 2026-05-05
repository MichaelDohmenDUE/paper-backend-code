import torch

from Utils.src.NodeLib.Node import PropsNode, Node, Graph, Signal
from Utils.src.RolloutBuffer import KStepRolloutBuffer
from backend.CommonModels.src.Policy_Reinforce_Baseline import PolicyReinforceBaseline
from backend.Utils.src.NodeLib.NodeLibrary import detransition, optimizer_update, policy_loss, mean_squared_error, \
    combined_loss, bellman


# --- Helper Functions for the Nodes ---

def extract_final_states(next_state, num_envs):
    final_next_states = next_state[-num_envs:]
    if final_next_states.dim() == 1:
        final_next_states = final_next_states.unsqueeze(0)
    return final_next_states


def compute_n_step_returns(rewards, dones, next_value_final, gamma):
    """Calculates G by iterating backwards through the N-step rollout."""
    returns = torch.zeros_like(rewards)
    G = next_value_final
    for t in reversed(range(len(rewards))):
        G = rewards[t] + gamma * (1.0 - dones[t]) * G
        returns[t] = G
    return returns


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

            Node("Detransition", ["spec_fields", "rollout", "device"],
                 ["state", "logps", "rewards", "dones", "next_state"],
                 function=detransition, no_grad=False),

            Node("AgentForward", ["behaviour", "state"], ["_logits", "value"],
                 function=lambda net, s: net(s), no_grad=False),

            Node("FlattenValue", ["value"], ["value_flat"],
                 function=lambda v: v.view(-1), no_grad=False),

            Node("ExtractFinalState", ["next_state", "num_envs"], ["final_next_states"],
                 function=extract_final_states, no_grad=True),

            Node("BootstrapForward", ["behaviour", "final_next_states"], ["_nl", "next_value_final"],
                 function=lambda net, s: net(s), no_grad=True),

            Node("FlattenBootstrap", ["next_value_final"], ["next_value_flat"],
                 function=lambda v: v.view(-1), no_grad=True),

            Node("ComputeReturns", ["rewards", "dones", "next_value_flat", "gamma"], ["G"],
                 function=compute_n_step_returns, no_grad=True),

            Node("ComputeAdvantage", ["G", "value_flat"], ["advantage"],
                 function=lambda g, v: g - v.detach(), no_grad=True),

            Node("PolicyLoss", ["logps", "advantage"], ["loss_policy"],
                 function=policy_loss, no_grad=False),

            Node("ValueLoss", ["value_flat", "G"], ["loss_value"],
                 function=mean_squared_error, no_grad=False),

            Node("TotalLoss", ["loss_policy", "c_pol", "loss_value", "c_val"], ["loss"],
                 function=combined_loss, no_grad=False),
            Node("TrainStep", ["optimizer", "loss"], ["grad_metrics"],
                 function=optimizer_update, no_grad=False)
        ]

        self.graph = Graph(nodes, initial_keys=list(self.context.keys()))

    def run(self):
        self.graph.run(self.context)

        #Logging'
        #Check if there is anything to Extract
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
