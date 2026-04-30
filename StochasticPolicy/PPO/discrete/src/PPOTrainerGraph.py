import numpy as np
import torch
from torch import nn

from backend.Utils.src.NodeLib.NodeLibrary import detransition, td_residual, normalize, clipped_surrogate_objective, \
    optimizer_normalized
from backend.Utils.src.RolloutBuffer import RolloutBuffer
from backend.Utils.src.utils import gae

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from backend.Utils.src.NodeLib.Node import Node, Graph, Signal


def create_ppo_minibatch_graph(actor, optimizer):
    nodes = [
        Node("AgentForward", ["actor", "b_states"], ["dist", "value_t"],
             function=lambda net, s: net(s)),
        Node("LogProb", ["dist", "b_actions"], ["new_logp"],
             function=lambda dist, a: dist.log_prob(a)),
        Node("Entropy", ["dist"], ["entropy"],
             function=lambda dist: dist.entropy().mean()),
        Node("SqueezeValue", ["value_t"], ["value_pred"],
             function=lambda v: v.squeeze(-1)),
        Node("PolicyLoss", ["new_logp", "b_old_logps", "b_adv", "clip_eps"], ["policy_loss"],
             function=clipped_surrogate_objective),
        Node("ValueLoss", ["b_ret", "value_pred"], ["value_loss"],
             function=lambda ret, v: 0.5 * (ret - v).pow(2).mean()),
        Node("TotalLoss", ["policy_loss", "value_loss", "entropy", "vf_coef", "ent_coef"], ["loss"],
             function=lambda policy_loss, value_loss, ent, value_coef,
                             entropy_coef: policy_loss + value_coef * value_loss - entropy_coef * ent),
        Node("Optimize", ["actor", "optimizer", "loss", "max_grad_norm"], ["_loss_val"],
             function=optimizer_normalized)
    ]

    initial_keys = [
        "actor", "optimizer", "b_states", "b_actions", "b_old_logps",
        "b_adv", "b_ret", "clip_eps", "vf_coef", "ent_coef", "max_grad_norm"
    ]
    return Graph(nodes, initial_keys=initial_keys)


class KUpdateNode(Node):
    def __init__(self, inner_graph, epochs, batch_size):
        super().__init__("PPOUpdateLoop",
                         ["states", "actions", "logps", "advantages", "returns", "inner_context"],
                         ["train_metrics"])
        self.inner_graph = inner_graph
        self.epochs = epochs
        self.batch_size = batch_size

    def forward(self, states, actions, logps, advantages, returns, context):
        dataset_size = len(states)
        indices = np.arange(dataset_size)
        p_losses, v_losses, entropies = [], [], []

        for _ in range(self.epochs):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, self.batch_size):
                idx = indices[start: start + self.batch_size]

                #Constructing the Inner graph needs context, copy enusres a clean start
                inner_context = context.copy()
                inner_context.update({
                    "b_states": states[idx],
                    "b_actions": actions[idx],
                    "b_old_logps": logps[idx],
                    "b_adv": advantages[idx],
                    "b_ret": returns[idx]
                })

                self.inner_graph.run(inner_context)
                # Logging Losses
                p_losses.append(inner_context["policy_loss"].item())
                v_losses.append(inner_context["value_loss"].item())
                entropies.append(inner_context["entropy"].item())

        return {
            "losses/policy_loss": np.mean(p_losses),
            "losses/value_loss": np.mean(v_losses),
            "losses/entropy": np.mean(entropies)
        }

class PPOTrainerProcessor:
    def __init__(self, actor, optimizer, rollout_buffer, batch_size=64, epochs=10,
                 clip_eps=0.2, vf_coef=1.0, ent_coef=0.01, max_grad_norm=0.5, gamma=0.99, lam=0.95):
        self.agent = actor
        self.rollout_buffer = rollout_buffer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.context = {
            "actor": actor, "optimizer": optimizer, "buffer": rollout_buffer,
            "device": self.device, "gamma": gamma, "lam": lam,
            "num_envs": rollout_buffer.num_envs, "fields": rollout_buffer.spec.fields,
            "inner_context": {
                "actor": actor, "optimizer": optimizer, "clip_eps": clip_eps,
                "vf_coef": vf_coef, "ent_coef": ent_coef, "max_grad_norm": max_grad_norm
            }
        }

        minibatch_graph = create_ppo_minibatch_graph(actor, optimizer)

        nodes = [
            Node("Sample", ["buffer"],["rollout"],
                 function=lambda b: b.sample() if b.reached_rollout_size() else Signal.NOSIGNAL),
            Node("Detransition", ["fields", "rollout", "device"],
                 ["states", "actions", "logps", "rewards", "dones", "values", "bootstraps"],
                 function=detransition),

            Node("GAE", ["rewards", "dones", "values", "bootstraps", "gamma", "lam", "num_envs"],
                 ["advantages", "returns"], function=self._gae_helper),

            KUpdateNode(minibatch_graph, epochs, batch_size)]

        self.graph = Graph(nodes, initial_keys=list(self.context.keys()))

    @staticmethod
    def _gae_helper(r, d, v, boot_vals, gamma, lam, num_envs):
        r, d, v, boot_vals = [x.view(-1, num_envs) for x in [r, d, v, boot_vals]]
        all_advantages = torch.zeros_like(v)

        for i in range(num_envs):
            boot_value = boot_vals[-1, i].unsqueeze(0)
            deltas = td_residual(r[:, i], d[:, i], v[:, i], boot_value, gamma)
            gae_adv = torch.zeros_like(deltas)
            last_gae = 0
            for t in reversed(range(len(deltas))):
                gae_adv[t] = last_gae = deltas[t] + gamma * lam * (1.0 - d[t, i]) * last_gae
            all_advantages[:, i] = gae_adv

        advantages = all_advantages.reshape(-1)
        returns = advantages + v.reshape(-1)
        return normalize(advantages), returns

    def run(self):
        self.graph.run(self.context)
        return self.context.get("train_metrics", {})