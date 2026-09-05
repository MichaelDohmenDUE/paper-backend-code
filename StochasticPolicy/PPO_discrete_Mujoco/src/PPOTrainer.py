import numpy as np
import torch

from Utils.src.NodeLib.NodeLibrary import detransition, normalize, clipped_surrogate_objective, td_residual, \
    compute_raw_gae, compute_returns, RepeatNode, record_metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from Utils.src.NodeLib.Node import Node, Graph, Signal, PropsNode


def dual_optimizer_step(actor, critic, optimizer, loss, max_norm):
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm)
    torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm)
    optimizer.step()
    return loss.item()


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

        Node("ActorForward", ["actor", "b_states"], ["dist"],
             function=lambda net, s: net(s)),
        Node("CriticForward", ["critic", "b_states"], ["value_tensor"],
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
             function=lambda ret, v: 0.5 * (ret - v).pow(2).mean()),
        Node("TotalLoss", ["policy_loss", "value_loss", "entropy", "vf_coef", "ent_coef"], ["loss"],
             function=lambda policy_loss, value_loss, ent, value_coef,
                             entropy_coef: policy_loss + value_coef * value_loss - entropy_coef * ent),
        Node("Optimize", ["actor", "critic", "optimizer", "loss", "max_grad_norm"], ["_loss_val"],
             function=dual_optimizer_step),

        # LoggingNode
        Node("RecordMetrics", ["metric_history", "policy_loss", "value_loss", "entropy"],
             ["metric_history"],
             function=record_metrics)
    ]

    initial_keys = [
        "actor", "critic", "optimizer", "clip_eps", "vf_coef", "ent_coef", "max_grad_norm", "_iteration",
        "all_batches", "metric_history"
    ]
    return Graph(nodes, initial_keys=initial_keys)


class PPOTrainerProcessor:
    def __init__(self, actor, critic, optimizer, rollout_buffer, replay_buffer, batch_size=64, epochs=10,
                 clip_eps=0.2, vf_coef=1.0, ent_coef=0.01, max_grad_norm=0.5, gamma=0.99, lam=0.95):
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
                "vf_coef": vf_coef, "ent_coef": ent_coef, "max_grad_norm": max_grad_norm, "device": self.device
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

            PropsNode("CriticForwardNext", ["next_state"], ["next_value"], props=["critic"],
                      function=lambda net, s: net(s), no_grad=True),

            PropsNode("td_residual", ["reward", "terminated", "value", "next_value", "gamma", "num_envs"],
                      ["deltas"],
                      function=td_residual),

            PropsNode("RawGAE", ["deltas", "terminated", "gamma", "lam", "num_envs"],
                      ["raw_advantages"],
                      function=compute_raw_gae),

            PropsNode("ComputeReturns", ["raw_advantages", "value"],
                      ["return"],
                      function=compute_returns),
            PropsNode("NormalizeAdvantages", ["raw_advantages"],
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
