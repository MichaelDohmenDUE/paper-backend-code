from torch.distributions import Categorical

from backend.Utils.src.BatchTransitioner import TransitionBatch
from backend.Utils.src.NodeLib.Node import Node
import torch
import torch.nn.utils as nn_utils

def rollout_to_batch():
    return Node(
        name="rollout_to_batch",
        function=lambda rollout, transition_spec: TransitionBatch(rollout, transition_spec).to_tensors(),
        inputs=["rollout", "transition_spec"],
        outputs=["batch"]
    )

def move_batch_to_device():
    return Node(
        name="move_batch_to_device",
        function=lambda batch, net: {
            k: (v.to(next(net.parameters()).device) if torch.is_tensor(v) else v)
            for k, v in batch.items()
        },
        inputs=["batch", "global_net"],
        outputs=["batch"]
    )

def unpack_a3c_batch():
    return Node(
        name="unpack_a3c_batch",
        function=lambda batch: (
            batch["reward"],
            batch["value"],
            batch["log_prob"],
            batch["done"],
            batch["entropy"],
        ),
        inputs=["batch"],
        outputs=["rewards", "values", "log_probs", "dones", "entropy"]
    )
def bootstrap_value():
    return Node(
        name="bootstrap_value",
        function=lambda last_done, last_value, global_net: (
            torch.zeros(1, device=next(global_net.parameters()).device)
            if last_done else last_value.detach().to(next(global_net.parameters()).device)
        ),
        inputs=["last_done", "last_value", "global_net"],
        outputs=["R_bootstrap"]
    )

def compute_returns():
    return Node(
        name="compute_returns",
        function=_compute_returns,
        inputs=["rewards", "dones", "gamma", "R_bootstrap", "global_net"],
        outputs=["returns"]
    )

def _compute_returns(rewards, dones, gamma, R_bootstrap, global_net):
    device = next(global_net.parameters()).device
    T_len = rewards.shape[0]
    returns = torch.zeros(T_len, device=device)
    R_t = R_bootstrap.to(device)
    for i in reversed(range(T_len)):
        R_t = rewards[i] + gamma * R_t * (1.0 - dones[i])
        returns[i] = R_t
    return returns

def bootstrap_value():
    return Node(
        name="bootstrap_value",
        function=lambda last_done, last_value, global_net: (
            torch.zeros(1, device=next(global_net.parameters()).device)
            if last_done else last_value.detach().to(next(global_net.parameters()).device)
        ),
        inputs=["last_done", "last_value", "global_net"],
        outputs=["R_bootstrap"]
    )

def compute_returns():
    return Node(
        name="compute_returns",
        function=_compute_returns,
        inputs=["rewards", "dones", "gamma", "R_bootstrap", "global_net"],
        outputs=["returns"]
    )

def _compute_returns(rewards, dones, gamma, R_bootstrap, global_net):
    device = next(global_net.parameters()).device
    T_len = rewards.shape[0]
    returns = torch.zeros(T_len, device=device)
    R_t = R_bootstrap.to(device)
    for i in reversed(range(T_len)):
        R_t = rewards[i] + gamma * R_t * (1.0 - dones[i])
        returns[i] = R_t
    return returns

def compute_advantages():
    return Node(
        name="compute_advantages",
        function=lambda returns, values: returns - values,
        inputs=["returns", "values"],
        outputs=["advantages"]
    )

def compute_policy_loss():
    return Node(
        name="compute_policy_loss",
        function=lambda log_probs, advantages: -(log_probs * advantages.detach()).sum(),
        inputs=["log_probs", "advantages"],
        outputs=["policy_loss"]
    )

def compute_value_loss():
    return Node(
        name="compute_value_loss",
        function=lambda advantages: advantages.pow(2).sum(),
        inputs=["advantages"],
        outputs=["value_loss"]
    )

def compute_entropy_term():
    return Node(
        name="compute_entropy_term",
        function=lambda entropy: entropy.sum(),
        inputs=["entropy"],
        outputs=["entropy_term"]
    )

def combine_losses():
    return Node(
        name="combine_losses",
        function=lambda policy_loss, value_loss, entropy_term, entropy_coef: (
            policy_loss + 0.5 * value_loss - entropy_coef * entropy_term
        ),
        inputs=["policy_loss", "value_loss", "entropy_term", "entropy_coef"],
        outputs=["loss"]
    )

def backward_on_local():
    return Node(
        name="backward_on_local",
        function=lambda local_net, optimizer, loss: (
            optimizer.zero_grad(),
            local_net.zero_grad(),
            loss.backward(),
        )[-1],
        inputs=["local_net", "optimizer", "loss"],
        outputs=["loss"]
    )

def clip_local_grads():
    return Node(
        name="clip_local_grads",
        function=lambda local_net, max_grad_norm: nn_utils.clip_grad_norm_(local_net.parameters(),
                                                                           max_grad_norm),
        inputs=["local_net", "max_grad_norm"],
        outputs=[]
    )

def push_local_grads_to_global():
    return Node(
        name="push_local_grads_to_global",
        function=_push_grads,
        inputs=["global_net", "local_net"],
        outputs=[]
    )

def _push_grads(global_net, local_net):
    for g, l in zip(global_net.parameters(), local_net.parameters()):
        if l.grad is None:
            continue
        if g.grad is None:
            g.grad = l.grad.detach().clone()
        else:
            g.grad.copy_(l.grad.detach())

def optimizer_step():
    return Node(
        name="optimizer_step",
        function=lambda optimizer, loss: (optimizer.step(), loss)[-1],
        inputs=["optimizer", "loss"],
        outputs=["loss"]
    )

def sync_local_with_global():
    return Node(
        name="sync_local_with_global",
        function=lambda global_net, local_net: local_net.load_state_dict(global_net.state_dict()),
        inputs=["global_net", "local_net"],
        outputs=[]
    )

def state_to_tensor():
    return Node(
        name="state_to_tensor",
        function=lambda state, device: torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0),
        inputs=["state", "device"],
        outputs=["state_t"]
    )

def forward_local_net():
    return Node(
        name="forward_local_net",
        function=lambda net, state_t: net(state_t),
        inputs=["local_net", "state_t"],
        outputs=["logits", "value"]
    )

def compute_action_distribution():
    return Node(
        name="compute_action_distribution",
        function=lambda logits: Categorical(torch.softmax(logits, dim=-1)),
        inputs=["logits"],
        outputs=["dist"]
    )

def sample_action():
    return Node(
        name="sample_action",
        function=lambda dist: dist.sample(),
        inputs=["dist"],
        outputs=["action"]
    )


def env_step():
    return Node(
        name="env_step",
        function=lambda env, action, episode_timesteps: env.step(action.item(), episode_timesteps=episode_timesteps),
        inputs=["env", "action", "episode_timesteps"],
        outputs=["next_state", "reward", "done", "info"]
    )

def build_transition():
    return Node(
        name="build_transition",
        function=lambda factory, state_t, action, reward, value, dist, done: factory.create(
            state=state_t,
            action=action,
            reward=float(reward),
            value=value.squeeze(-1),
            log_prob=dist.log_prob(action),
            done=bool(done),
            entropy=dist.entropy()
        ),
        inputs=["factory", "state_t", "action", "reward", "value", "dist", "done"],
        outputs=["transition"]
    )

def append_transition():
    return Node(
        name="append_transition",
        function=lambda rollout, transition: rollout.append(transition),
        inputs=["rollout", "transition"],
        outputs=[]
    )

def update_episode_reward():
    return Node(
        name="update_episode_reward",
        function=lambda episode_reward, reward: episode_reward + reward,
        inputs=["episode_reward", "reward"],
        outputs=["episode_reward"]
    )

def update_state_done():
    return Node(
        name="update_state_done",
        function=lambda next_state, done: (next_state, done),
        inputs=["next_state", "done"],
        outputs=["state", "done"]
    )

def increment_timestep():
    return Node(
        name="increment_timestep",
        function=lambda t: t + 1,
        inputs=["t"],
        outputs=["t"]
    )

def check_episode_end():
    return Node(
        name="check_episode_end",
        function=lambda done, episode_reward, avg_reward, beta, episode_count: (
            beta * avg_reward + (1 - beta) * episode_reward if done else avg_reward,
            episode_count + 1 if done else episode_count,
            done
        ),
        inputs=["done", "episode_reward", "avg_reward", "beta", "episode_count"],
        outputs=["avg_reward", "episode_count", "should_reset"]
    )

def reset_episode_if_needed():
    return Node(
        name="reset_episode_if_needed",
        function=lambda should_reset, env: env.reset() if should_reset else None,
        inputs=["should_reset", "env"],
        outputs=["reset_state"]
    )


def build_a3c_graph():
    return [
        rollout_to_batch(),
        move_batch_to_device(),
        unpack_a3c_batch(),
        bootstrap_value(),
        compute_returns(),
        compute_advantages(),
        compute_policy_loss(),
        compute_value_loss(),
        compute_entropy_term(),
        combine_losses(),
        backward_on_local(),
        clip_local_grads(),
        push_local_grads_to_global(),
        optimizer_step(),
        sync_local_with_global(),
    ]

def build_a3c_rollout_graph():
    return [
        state_to_tensor(),
        forward_local_net(),
        compute_action_distribution(),
        sample_action(),
        env_step(),
        build_transition(),
        append_transition(),
        update_episode_reward(),
        update_state_done(),
        increment_timestep(),
        check_episode_end(),
        reset_episode_if_needed(),
    ]