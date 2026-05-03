from NodeLib.Node import Graph
from backend.Utils.src.GlobalCounter import GlobalCounter
from backend.Utils.src.NodeLib.NodeLibrary import *
from backend.Utils.src.ReplayBuffer import ReplayBuffer

class TrainProcessor:
    """
    Twin Delayed Deep Deterministic Policy Gradient (TD3)
    Paper: https://arxiv.org/abs/1802.09477
    """

    def __init__(self, actor: nn.Module, critic_1: nn.Module, critic_2: nn.Module,
                 optimizer_critic_1: torch.optim.Optimizer, optimizer_critic_2: torch.optim.Optimizer,
                 optimizer_actor: torch.optim.Optimizer,
                 actor_target: nn.Module, critic_target_1: nn.Module, critic_target_2: nn.Module,
                 replay_buffer: ReplayBuffer, global_counter: GlobalCounter, max_action: float, learning_rate: float,
                 noise_clip: float, policy_noise, start_timesteps=25000, synchro_frequency: int = 2,
                 discount_factor: float = 0.99, device: torch.device = torch.device("cpu")):

        self.global_counter = global_counter
        self.start_timesteps = start_timesteps
        self.actor = actor
        self.context = {
            "actor": actor,
            "critic_1": critic_1,
            "critic_2": critic_2,
            "optimizer_critic_1": optimizer_critic_1,
            "optimizer_critic_2": optimizer_critic_2,
            "optimizer_actor": optimizer_actor,
            "actor_target": actor_target,
            "critic_target_1": critic_target_1,
            "critic_target_2": critic_target_2,
            "buffer": replay_buffer,
            "spec_fields": replay_buffer.spec.fields,
            "global_counter": global_counter,
            "max_action": max_action,
            "noise_clip": noise_clip,
            "policy_noise": policy_noise,
            "syncro_frequency": synchro_frequency,
            "discount_factor": discount_factor,
            "device": device,
        }

        nodes = [

            PropsNode("GetStep", ["global_counter"], ["current_step"],
                      function=lambda gc: gc.get(), no_grad=True),
            PropsNode("SampleBatch", ["buffer"], ["batch"],
                      function=lambda b: b.sample_batch()),
            PropsNode("Detransition", ["spec_fields", "batch", "device"],
                      ["state", "action", "reward", "next_state", "done"],
                      function=detransition),

            TimerNode(
                name="PolicyUpdateTimer",
                timer_inputs=["current_step", "syncro_frequency"],
                data_inputs=["state"],
                outputs=["gated_state"]
            ),
            PropsNode("TargetActionFwd", ["actor_target", "next_state"], ["next_action_clean"],
                      function=lambda net, s: net(s), no_grad=True),
            PropsNode("TargetActionNoise", ["next_action_clean", "policy_noise", "noise_clip", "max_action"],
                      ["next_action_noisy"],
                      function=action_with_gaussian_noise, no_grad=True),
            PropsNode("TargetActionClip", ["next_action_noisy", "max_action"], ["next_action"],
                      function=clipper, no_grad=True),

            PropsNode("TargetQ1", ["critic_target_1", "next_state", "next_action"], ["t_q1"],
                      function=lambda net, s, a: net(s, a).squeeze(), no_grad=True),
            PropsNode("TargetQ2", ["critic_target_2", "next_state", "next_action"], ["t_q2"],
                      function=lambda net, s, a: net(s, a).squeeze(), no_grad=True),
            PropsNode("MinQ", ["t_q1", "t_q2"], ["min_t_q"],
                      function=lambda q1, q2: torch.min(q1, q2), no_grad=True),
            PropsNode("TargetBellman", ["min_t_q", "reward", "done", "discount_factor"], ["target_Q"],
                      function=bellman, no_grad=True),

            PropsNode("CurrentQ1", ["critic_1", "state", "action"], ["current_Q1"],
                      function=lambda net, s, a: net(s, a).squeeze()),
            PropsNode("CurrentQ2", ["critic_2", "state", "action"], ["current_Q2"],
                      function=lambda net, s, a: net(s, a).squeeze()),
            PropsNode("CriticLoss1", ["current_Q1", "target_Q"], ["critic_loss_1"],
                      function=mean_squared_error),
            PropsNode("CriticLoss2", ["current_Q2", "target_Q"], ["critic_loss_2"],
                      function=mean_squared_error),
            PropsNode("UpdateCritic1", ["optimizer_critic_1", "critic_loss_1"], ["_c1_opt"],
                      function=optimizer_update),
            PropsNode("UpdateCritic2", ["optimizer_critic_2", "critic_loss_2"], ["_c2_opt"],
                      function=optimizer_update),

            PropsNode("ActorForward", ["actor", "gated_state", "_c1_opt", "_c2_opt"], ["policy_action"],
                      function=lambda net, s, *_: net(s)),

            PropsNode("ActorQValue", ["critic_1", "gated_state", "policy_action"], ["actor_q_val"],
                      function=lambda net, s, a: net(s, a).squeeze()),

            PropsNode("ActorLoss", ["actor_q_val"], ["actor_loss"],
                      function=deterministic_policy_gradient),

            PropsNode("ActorUpdate", ["optimizer_actor", "actor_loss"], ["actor_loss_val"],
                      function=lambda opt, loss: optimizer_update(opt, loss) or loss.item()),

            PropsNode("FormatMetrics", ["critic_loss_1", "critic_loss_2", "actor_loss_val", "current_Q1"],
                      ["metrics"], function=self._format_metrics, no_grad=True),

        ]

        self.graph = Graph(nodes, initial_keys=list(self.context.keys()))

    @staticmethod
    def _format_metrics(c_loss1, c_loss2, a_loss_val, q1_vals):
        metrics = {
            "losses/critic_loss 1": c_loss1.item(),
            "losses/critic_loss 2": c_loss2.item(),
            "losses/qf1_values": q1_vals.mean().item()
        }
        if a_loss_val is not Signal.NOSIGNAL and a_loss_val is not None:
            metrics["losses/actor_loss"] = a_loss_val
        return metrics

    def run(self):
        if self.global_counter.get() >= self.start_timesteps:
            self.graph.run(self.context)
        metrics = self.context.get("metrics", {})

        if metrics is Signal.NOSIGNAL or not isinstance(metrics, dict):
            return {}

        return metrics