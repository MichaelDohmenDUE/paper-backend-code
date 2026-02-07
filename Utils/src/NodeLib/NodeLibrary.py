import torch
import torch.nn.functional as F

from backend.Utils.src.NodeLib.Node import Node


def bellman(target_Q: torch.Tensor, reward: torch.Tensor, done: torch.Tensor, discount_factor: float) -> torch.Tensor:
    valid_transition = 1.0 - done
    target_Q = reward + valid_transition * discount_factor * target_Q
    return target_Q

def optimizer_update(optimizer: torch.optim.Optimizer, loss: torch.Tensor) -> torch.Tensor:
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss



class NodeLibrary:

    @staticmethod
    def can_sample_batch():
        return Node(
            name="can_sample_batch",
            function=lambda buffer: len(buffer) >= buffer.batch_size,
            inputs=["buffer"],
            outputs=["can_sample"]
        )

    @staticmethod
    def sample_batch():
        return Node(
            name="sample_batch",
            function=lambda buffer: buffer.sample_batch(),
            inputs=["buffer"],
            outputs=["batch"]
        )

    @staticmethod
    def move_batch_to_device():
        return Node(
            name="move_batch_to_device",
            function=lambda batch, device: {k: v.to(device) for k, v in batch.items()},
            inputs=["batch", "device"],
            outputs=["batch"]
        )

    @staticmethod
    def unpack_batch():
        return Node(
            name="unpack_batch",
            function=lambda batch: (
                batch["state"],
                batch["action"],
                batch["reward"],
                batch["next_state"],
                batch["done"],
            ),
            inputs=["batch"],
            outputs=["states", "actions", "rewards", "next_states", "dones"]
        )

    @staticmethod
    def compute_q_values():
        return Node(
            name="compute_q_values",
            function=lambda net, states: net(states),
            inputs=["behavior_net", "states"],
            outputs=["q_values"]
        )

    @staticmethod
    def gather_qsa():
        return Node(
            name="gather_qsa",
            function=lambda q_values, actions: q_values.gather(1, actions.long()),
            inputs=["q_values", "actions"],
            outputs=["qsa_behavior"]
        )

    @staticmethod
    def compute_next_q_online():
        return Node(
            name="compute_next_q_online",
            function=lambda net, next_states: net(next_states),
            inputs=["behavior_net", "next_states"],
            outputs=["next_q_online"],
            no_grad=True
        )

    @staticmethod
    def compute_next_actions():
        return Node(
            name="compute_next_actions",
            function=lambda next_q_online: next_q_online.argmax(dim=1, keepdim=True),
            inputs=["next_q_online"],
            outputs=["next_actions"],
            no_grad=True
        )

    @staticmethod
    def compute_next_q_target():
        return Node(
            name="compute_next_q_target",
            function=lambda net, next_states: net(next_states),
            inputs=["target_net", "next_states"],
            outputs=["next_q_target"],
            no_grad=True
        )

    @staticmethod
    def compute_qsa_target():
        return Node(
            name="compute_qsa_target",
            function=lambda next_q_target, next_actions: next_q_target.gather(1, next_actions),
            inputs=["next_q_target", "next_actions"],
            outputs=["qsa_target"],
            no_grad=True
        )

    @staticmethod
    def compute_ddqn_target():
        return Node(
            name="compute_ddqn_target",
            function=lambda rewards, qsa_target, dones, gamma: rewards + gamma * qsa_target * (1.0 - dones),
            inputs=["rewards", "qsa_target", "dones", "gamma"],
            outputs=["target"],
            no_grad=True
        )

    @staticmethod
    def compute_dqn_target():
        return Node(
            name="compute_dqn_target",
            function=lambda rewards, next_q_target, dones, gamma: rewards + gamma * next_q_target.max(dim=1,
                                                                                                      keepdim=True).values * (
                                                                          1.0 - dones),
            inputs=["rewards", "next_q_target", "dones", "gamma"],
            outputs=["target"],
            no_grad=True
        )

    @staticmethod
    def compute_mse_loss():
        return Node(
            name="compute_mse_loss",
            function=lambda pred, target: F.mse_loss(pred, target),
            inputs=["qsa_behavior", "target"],
            outputs=["loss"]
        )

    @staticmethod
    def compute_huber_loss():
        return Node(
            name="compute_huber_loss",
            function=lambda pred, target: F.smooth_l1_loss(pred, target),
            inputs=["qsa_behavior", "target"],
            outputs=["loss"]
        )

    @staticmethod
    def optimizer_update():
        return Node(
            name="optimizer_update",
            function=optimizer_update,
            inputs=["optimizer", "loss"],
            outputs=["loss"]
        )

    @staticmethod
    def bellman():
        return Node(
            name="bellman",
            function=bellman,
            inputs=["target_q", "reward", "done", "discount_factor"],
            outputs=["target_q"]
        )
