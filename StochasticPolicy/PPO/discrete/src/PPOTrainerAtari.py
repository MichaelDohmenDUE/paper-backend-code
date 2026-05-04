import numpy as np
import torch
from torch import nn

from backend.Utils.src.NodeLib.NodeLibrary import mean_squared_error
from backend.Utils.src.NodeLib.NodeLibrary import detransition, normalize, clipped_surrogate_objective, \
    optimizer_normalized, td_residual, compute_returns, record_metrics, RepeatNode, compute_raw_gae
from backend.Utils.src.ReplayBuffer import ReplayBuffer
from backend.Utils.src.RolloutBuffer import RolloutBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from backend.Utils.src.NodeLib.Node import Node, Graph, Signal, PropsNode


def create_ppo_minibatch_graph():
    nodes = [

        Node("GetBatch", ["all_batches", "_iteration"],
             ["b_states", "b_action", "b_old_logp", "b_adv", "b_ret"],
             function=lambda batches, i: (
                 batches[i]["state"].to(device),
                 batches[i]["action"].to(device),
                 batches[i]["logp"].to(device),
                 batches[i]["advantage"].to(device),
                 batches[i]["return"].to(device)
             )),

        Node("AgentForward", ["agent", "b_states"], ["dist", "value_tensor"],
             function=lambda net, s: net(s)),
        Node("LogProb", ["dist", "b_action"], ["new_logp"],
             function=lambda dist, a: dist.log_prob(a)),
        Node("Entropy", ["dist"], ["entropy"],
             function=lambda dist: dist.entropy().mean()),
        Node("SqueezeValue", ["value_tensor"], ["value_pred"],
             function=lambda v: v.squeeze(-1)),
        Node("PolicyLoss", ["new_logp", "b_old_logp", "b_adv", "clip_eps"], ["policy_loss"],
             function=clipped_surrogate_objective),
        Node("ValueLoss", ["b_ret", "value_pred"], ["value_loss"],
             function=lambda ret, v: 0.5 * mean_squared_error(v, ret)),
        Node("TotalLoss", ["policy_loss", "value_loss", "entropy", "vf_coef", "ent_coef"], ["loss"],
             function=lambda policy_loss, value_loss, ent, value_coef,
                             entropy_coef: policy_loss + value_coef * value_loss - entropy_coef * ent),
        Node("Optimizer Normalized", ["agent", "optimizer", "loss", "max_grad_norm"], ["_loss_val"],
             function=optimizer_normalized),
        # LoggingNode
        Node("RecordMetrics", ["metric_history", "policy_loss", "value_loss", "entropy"],
             ["metric_history"],
             function=record_metrics)
    ]

    initial_keys = [
        "agent", "optimizer", "clip_eps", "vf_coef", "ent_coef", "max_grad_norm", "_iteration",
        "all_batches", "metric_history"
    ]
    return Graph(nodes, initial_keys=initial_keys)


class PPOTrainerProcessor:
    def __init__(self, agent: nn.Module, optimizer: torch.optim.Optimizer, rollout_buffer: RolloutBuffer,
                 replay_buffer: ReplayBuffer, batch_size=64,
                 epochs=10, clip_eps=0.2, vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5, gamma=0.99, lam=0.95):
        self.agent = agent
        self.rollout_buffer = rollout_buffer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.context = {
            "agent": agent, "optimizer": optimizer, "buffer": rollout_buffer, "replay_buffer": replay_buffer,
            "device": self.device, "gamma": gamma, "lam": lam,
            "num_envs": rollout_buffer.num_envs, "spec_fields": rollout_buffer.spec.fields,
            "inner_context": {
                "agent": agent, "optimizer": optimizer, "clip_eps": clip_eps,
                "vf_coef": vf_coef, "ent_coef": ent_coef, "max_grad_norm": max_grad_norm, "device": self.device,
            }
        }

        minibatch_graph = create_ppo_minibatch_graph()

        nodes = [
            PropsNode("Sample", ["buffer"], ["rollout"],
                      function=lambda b: b.sample() if b.reached_rollout_size() else Signal.NOSIGNAL),

            PropsNode("Detransition", ["spec_fields", "rollout", "device"],
                      ["state", "action", "logp", "reward", "done", "value", "bootstrap"],
                      function=detransition),
            PropsNode("td_residual", ["reward", "done", "value", "bootstrap", "gamma", "num_envs"],
                      ["deltas"],
                      function=td_residual),

            PropsNode("RawGAE", ["deltas", "done", "gamma", "lam", "num_envs"],
                      ["raw_advantages"],
                      function=compute_raw_gae),

            PropsNode("ComputeReturns", ["raw_advantages", "value"],
                      ["return"],
                      function=compute_returns),
            PropsNode("NormalizeAdvantages", ["raw_advantages"],
                      ["advantage"],
                      function=normalize),

            PropsNode("PopulateBuffer",
                      ["replay_buffer", "state", "action", "logp", "advantage", "return"],
                      ["ppo_buffer"],
                      function=lambda buffer, *args: buffer.populate(dict(zip(buffer.spec.fields, args)))),

            PropsNode("GenerateBatches",
                      ["ppo_buffer"],
                      ["all_batches"],
                      function=lambda buffer: buffer.generate_batches(batch_size, epochs)),

            PropsNode("PrepInnerContext", ["inner_context", "all_batches", "metric_history"],
                      ["ready_inner_context"],
                      function=lambda ctx, batches, hist: {**ctx, "all_batches": batches, "metric_history": hist}),

            RepeatNode("KUpdateLoop",
                       inputs=["ready_inner_context"],
                       outputs=["final_inner_context"],
                       inner_graph=minibatch_graph,
                       iterations=(rollout_buffer.rollout_size // batch_size) * epochs),

            PropsNode("InitMetrics", [], ["metric_history"],
                      function=lambda: {"policy_loss": [], "value_loss": [], "entropy": []}),

        ]

        self.graph = Graph(nodes, initial_keys=list(self.context.keys()))

    def run(self):
        self.graph.run(self.context)

        final_context = self.context.get("final_inner_context")
        if final_context and isinstance(final_context, dict):
            history = final_context.get("metric_history", {})
            train_metrics = {
                "losses/policy_loss": np.mean(history["policy_loss"]) if history.get("policy_loss") else 0,
                "losses/value_loss": np.mean(history["value_loss"]) if history.get("value_loss") else 0,
                "losses/entropy": np.mean(history["entropy"]) if history.get("entropy") else 0
            }
            return train_metrics

        return {}
