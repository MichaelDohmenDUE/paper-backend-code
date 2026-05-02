import torch

from NodeLib.NodeLibrary import combined_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch

from backend.Utils.src.NodeLib.NodeLibrary import detransition, normalize, clipped_surrogate_objective, td_residual, \
    compute_raw_gae, compute_returns, KUpdateNode

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
        PropsNode("ActorForward", ["b_states"], ["dist"], ["actor"],
                  function=lambda net, s: net(s)),
        Node("CriticForward", ["critic", "b_states"], ["value_tensor"],
             function=lambda net, s: net(s)),
        Node("LogProb", ["dist", "b_actions"], ["new_logp"],
             function=lambda dist, a: dist.log_prob(a).sum(dim=-1) if len(
                 dist.log_prob(a).shape) > 1 else dist.log_prob(a)),
        Node("SqueezeValue", ["value_tensor"], ["value_pred"],
             function=lambda v: v.squeeze(-1)),
        Node("PolicyLoss", ["new_logp", "b_old_logps", "b_adv", "clip_eps"], ["policy_loss"],
             function=clipped_surrogate_objective),
        Node("ValueLoss", ["b_ret", "value_pred"], ["value_loss"],
             function=lambda ret, v: 0.5 * (ret - v).pow(2).mean()),
        Node("TotalLoss", ["policy_loss", "value_loss", "entropy", "vf_coef", "ent_coef"], ["loss"],
             function=combined_loss),
        PropsNode("DualOptimizer", ["loss", "max_grad_norm"], ["_loss_val"], ["actor", "critic", "optimizer"],
                  function=dual_optimizer_step)
    ]

    initial_keys = [
        "actor", "critic", "optimizer",
        "b_states", "b_actions", "b_old_logps",
        "b_adv", "b_ret", "clip_eps", "vf_coef", "ent_coef", "max_grad_norm"
    ]
    return Graph(nodes, initial_keys=initial_keys)


class PPOTrainerProcessor:
    def __init__(self, actor, critic, optimizer, rollout_buffer, batch_size=64, epochs=10,
                 clip_eps=0.2, vf_coef=1.0, ent_coef=0.01, max_grad_norm=0.5, gamma=0.99, lam=0.95):
        self.actor = actor
        self.critic = critic
        self.rollout_buffer = rollout_buffer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.context = {
            "actor": actor, "critic": critic, "optimizer": optimizer, "buffer": rollout_buffer,
            "device": self.device, "gamma": gamma, "lam": lam,
            "num_envs": rollout_buffer.num_envs, "fields": rollout_buffer.spec.fields,
            "inner_context": {
                "actor": actor, "critic": critic, "optimizer": optimizer, "clip_eps": clip_eps,
                "vf_coef": vf_coef, "ent_coef": ent_coef, "max_grad_norm": max_grad_norm
            }
        }

        minibatch_graph = create_ppo_minibatch_graph()

        nodes = [
            PropNode("Sample", ["buffer"], ["rollout"],
                     function=lambda b: b.sample() if b.reached_rollout_size() else Signal.NOSIGNAL),
            Node("Detransition", ["fields", "rollout", "device"],
                 ["states", "actions", "logps", "rewards", "dones", "values", "bootstraps"],
                 function=detransition),

            Node("td_residual", ["rewards", "dones", "values", "bootstraps", "gamma", "num_envs"],
                 ["deltas"],
                 function=td_residual),

            Node("RawGAE", ["deltas", "dones", "gamma", "lam", "num_envs"],
                 ["raw_advantages"],
                 function=compute_raw_gae),

            Node("ComputeReturns", ["raw_advantages", "values"],
                 ["returns"],
                 function=compute_returns),
            Node("NormalizeAdvantages", ["raw_advantages"],
                 ["advantages"],
                 function=normalize),

            KUpdateNode(minibatch_graph, epochs, batch_size)]

        self.graph = Graph(nodes, initial_keys=list(self.context.keys()))

    def run(self):
        self.graph.run(self.context)
        return self.context.get("train_metrics", {})
