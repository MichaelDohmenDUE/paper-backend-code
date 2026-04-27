from xxlimited_35 import Null

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical

from backend.Utils.src.NodeLib.Node import Node
from backend.Utils.src.ReplayBuffer import ReplayBuffer

def action_with_gaussian_noise(action, policy_noise, noise_clip, max_action):
    noise = (torch.randn_like(action) * policy_noise).clamp(-noise_clip, noise_clip)
    action_noisy = (action + noise)
    return action_noisy

def clipper(action, max_action):
    return action.clamp(-max_action, max_action)

def bellman(target_Q: torch.Tensor, reward: torch.Tensor, done: torch.Tensor, discount_factor: float) -> torch.Tensor:
    valid_transition = 1.0 - done
    target_Q = reward + valid_transition * discount_factor * target_Q
    return target_Q

def reset_handler(env, next_state, done):
    state = next_state
    if done:
        state = env.reset()
        return state
    return state

def categorical_distribution(logits: torch.Tensor) -> Categorical:
    return torch.distributions.Categorical(logits=logits)

def sample_distribution(dist: Categorical) -> tuple[int, torch.Tensor]:
    action_dist = dist.sample()
    log_prob = dist.log_prob(action_dist).squeeze(0)
    return int(action_dist.item()), log_prob

def optimizer_update(optimizer: torch.optim.Optimizer, loss: torch.Tensor) -> torch.Tensor:
    optimizer.zero_grad()
    loss.backward()
    grad_metrics = {}
    for group in optimizer.param_groups:
        for p in group['params']:
            if p.grad is not None:
                param_id = str(p.shape)
                grad_data = p.grad.detach()
                grNone:ad_metrics[f"grad_norm_{param_id}"] = grad_data.norm(2).item()
                grad_metrics[f"grad_std_{param_id}"] = grad_data.std().item()
    optimizer.step()
    return grad_metrics

def timed_optimizer_update(optim: torch.optim.Optimizer, loss: torch.Tensor, step: int, syncro_frequency: int):
    if step % syncro_frequency != 0:
        return None
    optimizer_update(optimizer=optim, loss=loss)
    return loss.item()

def deterministic_policy_gradient(values: torch.Tensor) -> torch.Tensor:
    return -values.mean()

def policy_loss(tensor_1: torch.Tensor, tensor_2: torch.Tensor) -> torch.Tensor:
    return -(tensor_1 * tensor_2).sum()

def argmax(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.argmax(dim=1, keepdim=True)

def indexing(tensor: torch.Tensor, index) -> torch.Tensor:
    return tensor.gather(1, index)


def nl_max(tensor: torch.Tensor, dim: int = 1) -> torch.Tensor:
    return tensor.max(dim=dim, keepdim=True).values

def mean_squared_error(tensor: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(tensor, target)

def combined_loss(*args):
    if len(args) % 2 == 1:
        print("Error in combining losses, you likely missed a loss or scalar")
        return None
    return sum(loss * weight for loss, weight in zip(args[0::2], args[1::2]))


def action_with_noise(noise,  action_tensor: torch.Tensor, max_action) -> torch.Tensor:
    noise = noise.sample()
    action_tensor = action_tensor + noise
    action_tensor = action_tensor.clamp(-max_action, max_action)
    return action_tensor

def noise_handler(noise, done: bool):
    if done:
        noise.reset()


def optimizer_normalized(net: nn.Module, optimizer: torch.optim.Optimizer, loss: torch.Tensor,
                         max_norm: float) -> torch.Tensor:
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm)
    optimizer.step()
    return loss

def td_residual(rewards, dones, value, bootstrap_value, gamma):
    next_values = torch.zeros_like(value)
    next_values[:-1] = value[1:]
    next_values[-1] = bootstrap_value[-1]

    next_non_terminal = 1.0 - dones

    assert (bootstrap_value[:-1] == 0.0).all(), "bootstrap_value Error"

    deltas = rewards + gamma * next_values * next_non_terminal - value
    return deltas

def normalize(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.numel() <= 1:
        return tensor
    return (tensor - tensor.mean()) / (tensor.std() + 1e-8)

def detransition(fields, batch, device: torch.device):
    processed = {}

    for key, tensor in batch.items():
        t = tensor.to(device)
        if "state" in key:
            if t.dtype == torch.uint8:
                processed[key] = t.contiguous().float() / 255.0
            else:
                processed[key] = t.float()
        elif "action" in key:
            is_discrete = not t.is_floating_point() or torch.all(t == t.long())
            processed[key] = t.long() if is_discrete else t.float()
        elif key in ["reward", "done"]:
            processed[key] = t.float().flatten()

        else:
            processed[key] = t

    return tuple(processed[k] for k in fields)

def clipped_surrogate_objective(new_logp, old_logps, advantage, clip_eps):
    ratio = torch.exp(new_logp - old_logps)
    surrogate_objective = ratio * advantage
    surrogate_objective2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantage
    policy_loss = -torch.min(surrogate_objective, surrogate_objective2).mean()
    return policy_loss

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
