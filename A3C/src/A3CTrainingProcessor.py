import torch
import torch.nn.utils as nn_utils

from backend.Utils.src.BatchTransitioner import TransitionSpec, TransitionBatch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class A3CTrainingProcessor:
    def __init__(self, global_net, local_net, optimizer, gamma, entropy_coef=0.001, max_grad_norm=40.0):
        self.global_net = global_net
        self.local_net = local_net
        self.optimizer = optimizer
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

    def run(self, rollout):
        if len(rollout) == 0:
            return

        last_tr = rollout[-1]

        if last_tr.done:
            R = torch.zeros(1, device=device)
        else:
            R = last_tr.value.detach()  # V(s_t, θ′_v)

        batch = TransitionBatch(rollout, TransitionSpec(["state", "action", "reward", "value", "log_prob", "done", "entropy"]))
        t = batch.to_tensors()

        rewards = t["reward"]
        values = t["value"]
        log_probs = t["log_prob"]
        dones = t["done"]
        entropy = t["entropy"]


        T_len = rewards.shape[0]
        returns = torch.zeros(T_len, device=device)
        R_t = R
        for i in reversed(range(T_len)):
            R_t = rewards[i] + self.gamma * R_t * (1.0 - dones[i])
            returns[i] = R_t

        advantages = returns - values


        policy_loss = -(log_probs * advantages.detach()).sum()
        value_loss = advantages.pow(2).sum()
        entropy_term = entropy.sum()

        loss = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy_term


        self.optimizer.zero_grad()
        self.local_net.zero_grad()
        loss.backward()

        nn_utils.clip_grad_norm_(self.local_net.parameters(), self.max_grad_norm)

        for global_param, local_param in zip(self.global_net.parameters(), self.local_net.parameters()):
            if global_param.grad is None:
                global_param.grad = local_param.grad.detach().clone()
            else:
                global_param.grad.copy_(local_param.grad.detach())

        self.optimizer.step()

        self.local_net.load_state_dict(self.global_net.state_dict())
