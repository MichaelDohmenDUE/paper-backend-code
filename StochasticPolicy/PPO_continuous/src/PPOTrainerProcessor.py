import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch

from backend.Utils.src.NodeLib.NodeLibrary import detransition, normalize, clipped_surrogate_objective, td_residual, \
    compute_raw_gae, compute_returns, record_metrics, RepeatNode

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from backend.Utils.src.NodeLib.Node import Node, Graph, Signal, PropsNode


def dual_optimizer_step(actor, critic, optimizer, loss, max_norm):
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm)
    torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm)
    optimizer.step()
    return loss.item()


def create_ppo_minibatch_graph():
    nodes = [

        PropsNode("GetBatch", ["all_batches", "_iteration"],
                  ["b_state", "b_action", "b_old_logps", "b_adv", "b_ret"],
                  function=lambda batches, i: (
                      batches[i]["state"].to(device),
                      batches[i]["action"].to(device),
                      batches[i]["logp"].to(device),
                      batches[i]["advantage"].to(device),
                      batches[i]["return"].to(device)
                  )),

        PropsNode("ActorForward", ["b_state"], ["dist"], props=["actor"],
                  function=lambda net, s: net(s)),
        PropsNode("CriticForward", ["b_state"], ["value_tensor"], props=["critic"],
                  function=lambda net, s: net(s)),
        PropsNode("LogProb", ["dist", "b_action"], ["new_logp"],
                  function=lambda dist, a: dist.log_prob(a).sum(dim=-1) if len(
                      dist.log_prob(a).shape) > 1 else dist.log_prob(a)),
        PropsNode("Entropy", ["dist"], ["entropy"],
                  function=lambda dist: dist.entropy().mean()),

        PropsNode("SqueezeValue", ["value_tensor"], ["value_pred"],
                  function=lambda v: v.squeeze(-1)),
        PropsNode("PolicyLoss", ["new_logp", "b_old_logps", "b_adv", "clip_eps"], ["policy_loss"],
                  function=clipped_surrogate_objective),
        PropsNode("ValueLoss", ["b_ret", "value_pred"], ["value_loss"],
                  function=lambda ret, v: 0.5 * (ret - v).pow(2).mean()),
        PropsNode("TotalLoss", ["policy_loss", "value_loss", "entropy", "vf_coef", "ent_coef"], ["loss"],
                  function=lambda policy_loss, value_loss, ent, value_coef, entropy_coef:
                  policy_loss + value_coef * value_loss - entropy_coef * ent),
        PropsNode("DualOptimizer", ["loss", "max_grad_norm"], ["_loss_val"], ["actor", "critic", "optimizer"],
                  function=dual_optimizer_step),
        PropsNode("RecordMetrics", ["metric_history", "policy_loss", "value_loss", "entropy"],
                  ["metric_history"],
                  function=record_metrics)
    ]

    initial_keys = [
        "actor", "critic", "optimizer", "all_batches", "clip_eps", "vf_coef", "pl_coef", "ent_coef", "max_grad_norm",
        "_iteration", "metric_history",
    ]
    return Graph(nodes, initial_keys=initial_keys)


class PPOTrainerProcessor:
    def __init__(self, actor, critic, optimizer, rollout_buffer, replay_buffer, batch_size=64, epochs=10,
                 clip_eps=0.2, vf_coef=0.5, pl_coef=1.0, ent_coef=0.00, max_grad_norm=0.5, gamma=0.99, lam=0.95):
        self.actor = actor
        self.critic = critic
        self.rollout_buffer = rollout_buffer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.context = {
            "actor": actor, "critic": critic, "optimizer": optimizer, "buffer": rollout_buffer,
            "replay_buffer": replay_buffer,
            "device": self.device, "gamma": gamma, "lam": lam,
            "num_envs": rollout_buffer.num_envs, "spec_fields": rollout_buffer.spec.fields,
            "inner_context": {
                "actor": actor, "critic": critic, "optimizer": optimizer, "clip_eps": clip_eps,
                "vf_coef": vf_coef, "pl_coef": pl_coef, "max_grad_norm": max_grad_norm, "ent_coef": ent_coef,
            }
        }

        minibatch_graph = create_ppo_minibatch_graph()

        nodes = [
            PropsNode("Sample", ["buffer"], ["rollout"],
                      function=lambda b: b.sample() if b.reached_rollout_size() else Signal.NOSIGNAL),
            PropsNode("Detransition", ["spec_fields", "rollout", "device"],
                      ["state", "action", "logp", "reward", "terminated", "next_state"],
                      function=detransition),

            PropsNode("CriticForward", ["state"], ["value"], props=["critic"],
                      function=lambda net, s: net(s), no_grad=True),

            PropsNode("CriticForwardNxt", ["next_state"], ["next_value"], props=["critic"],
                      function=lambda net, s: net(s), no_grad=True),

            PropsNode("td_residual", ["reward", "terminated", "value", "next_value", "gamma", "num_envs"],
                      ["deltas"],
                      function=td_residual),

            Node("RawGAE", ["deltas", "terminated", "gamma", "lam", "num_envs"],
                 ["raw_advantages"],
                 function=compute_raw_gae),

            Node("ComputeReturns", ["raw_advantages", "value"],
                 ["return"],
                 function=compute_returns),
            Node("NormalizeAdvantages", ["raw_advantages"],
                 ["advantage"],
                 function=normalize),

            PropsNode("PopulateBuffer",
                      ["state", "action", "logp", "advantage", "return"],
                      ["ppo_buffer"], props=["replay_buffer"],
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
        final_inner = self.context.get("final_inner_context")
        if final_inner is not None and final_inner is not Signal.NOSIGNAL:
            try:
                history = final_inner["metric_history"]
                train_metrics = {
                    "losses/policy_loss": np.mean(history["policy_loss"]) if history["policy_loss"] else 0,
                    "losses/value_loss": np.mean(history["value_loss"]) if history["value_loss"] else 0,
                    "losses/entropy": np.mean(history["entropy"]) if history["entropy"] else 0
                }
                return train_metrics
            except (KeyError, TypeError):
                return {}
        return {}
