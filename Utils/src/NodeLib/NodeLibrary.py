import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical

from NodeLib.Node import PropsNode, Signal
from backend.Utils.src.NodeLib.Node import Node
from backend.Utils.src.RolloutBuffer import RolloutBuffer


def record_metrics(history, p_loss, v_loss, entropy):
    """Appends the current step's losses to the tracking dictionary."""
    history["policy_loss"].append(p_loss.item())
    history["value_loss"].append(v_loss.item())
    history["entropy"].append(entropy.item())
    return history

def action_with_gaussian_noise(action, policy_noise, noise_clip, max_action):
    noise = (torch.randn_like(action) * policy_noise).clamp(-noise_clip, noise_clip)
    action_noisy = (action + noise)
    return action_noisy


def to_tensor(state, device: torch.device, dtype=torch.float32) -> torch.Tensor:
    return torch.as_tensor(state, dtype=dtype, device=device)


def to_numpy_array(action_raw):
    if torch.is_tensor(action_raw):
        a = action_raw.detach().cpu().numpy()
    else:
        a = action_raw
    return np.atleast_1d(a)


def clipper(action, max_action):
    if isinstance(action, torch.Tensor):
        return action.clamp(-max_action, max_action)
    else:
        return np.clip(action, -max_action, max_action)


def bellman(target_Q: torch.Tensor, reward: torch.Tensor, done: torch.Tensor, discount_factor: float) -> torch.Tensor:
    valid_transition = 1.0 - done
    target_Q = reward + valid_transition * discount_factor * target_Q
    return target_Q


def soft_bellman(target_Q, reward: torch.Tensor, done: torch.Tensor, discount_factor: float,
                 temp: torch.Tensor, logp: torch.Tensor) -> torch.Tensor:
    valid_transition = 1.0 - done
    return reward + discount_factor * valid_transition * (target_Q - temp * logp)


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


def optimizer_update(optimizer: torch.optim.Optimizer, loss: torch.Tensor) -> dict:
    optimizer.zero_grad()
    loss.backward()
    #total_squared_norm = 0.0
    #n_params = 0
    #for group in optimizer.param_groups:
    #    for p in group['params']:
     ##       if p.grad is not None:
     #           total_squared_norm += p.grad.detach().pow(2).sum().item()
     #           n_params += p.numel()
    #rms_grad = (total_squared_norm / n_params) ** 0.5 if n_params > 0 else 0
    optimizer.step()
    return {}
    return {
        "grad/rms_gradient": rms_grad,
        "grad/total_parameters": n_params
    }


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


def action_with_noise(noise, action_tensor: torch.Tensor, max_action) -> torch.Tensor:
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


def optimize_step_normalized(optimizer, actor, loss, max_grad_norm):  # TODO Unify this later , looks up
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
    optimizer.step()
    return True


def td_residual(rewards, dones, values, bootstrap_values, gamma, num_envs=None):
    if num_envs is not None:
        rewards = rewards.view(-1, num_envs)
        dones = dones.view(-1, num_envs)
        values = values.view(-1, num_envs)
        bootstrap_values = bootstrap_values.view(-1, num_envs)
    next_non_terminal = 1.0 - dones
    assert (bootstrap_values[:-1] == 0.0).all(), "bootstrap_value Error"
    # Shift values by 1 and Bootstrap the bootstrapValue
    next_values = torch.cat([values[1:], bootstrap_values[-1:]], dim=0)
    deltas = rewards + gamma * next_values * next_non_terminal - values

    return deltas

def compute_raw_gae(deltas, d, gamma, lam, num_envs):
    d = d.view(-1, num_envs)
    all_advantages = torch.zeros_like(deltas)

    last_gae = torch.zeros(num_envs, device=deltas.device)

    for t in reversed(range(len(deltas))):
        last_gae = deltas[t] + gamma * lam * (1.0 - d[t]) * last_gae
        all_advantages[t] = last_gae

    return all_advantages.reshape(-1)


def compute_returns(advantages, values):
    return advantages + values.view(-1)


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
                processed[key] = t.contiguous().float()  # / 255.0
            else:
                processed[key] = t.float()
        elif "action" in key:
            is_discrete = not t.is_floating_point() or torch.all(t == t.long())
            processed[key] = t.long() if is_discrete else t.float()
        elif key in ["reward", "done"]:
            processed[key] = t.float().view(-1)

        else:
            processed[key] = t

    return tuple(processed[k] for k in fields)

def unpack_minibatch(minibatch_dict, fields):
    return tuple(minibatch_dict[k] for k in fields)

def clipped_surrogate_objective(new_logp, old_logps, advantage, clip_eps):
    ratio = torch.exp(new_logp - old_logps)
    surrogate_objective = ratio * advantage
    surrogate_objective2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantage
    policy_loss = -torch.min(surrogate_objective, surrogate_objective2).mean()
    return policy_loss


class TransitionNode(Node):
    def __init__(self, factory, input_mapping: dict[str, str], default_kwargs: dict = None):
        self.factory_args = list(input_mapping.keys())
        context_keys = list(input_mapping.values())

        super().__init__("Transition", context_keys + ["num_envs"], ["transitions"])

        self.factory = factory
        self.default_kwargs = default_kwargs if default_kwargs is not None else {}

    def forward(self, *args):
        *transition_data, num_envs = args

        transitions = []

        for step_values in zip(*transition_data):
            kwargs = dict(zip(self.factory_args, step_values))
            # Inject the hardcoded values
            kwargs.update(self.default_kwargs)
            transition = self.factory.forward(**kwargs)
            transitions.append(transition)

        return transitions


class BufferAppendingNode(Node):
    def __init__(self):
        super().__init__("BufferAppendingNode", ["buffer", "transitions"], ["_buffer_updated"], ["buffer",])

    def forward(self, buffer, transitions):
        #print(transitions)
        for t in transitions:
            buffer.append(t)
        return True  # DummySignal


class BootStrappingNode(Node):
    def __init__(self, rollout_size):
        super().__init__("Bootstrapping",
                         ["buffer", "next_state", "done", "agent", "device", "_buffer_updated"],
                         [], no_grad=True)
        self.rollout_size = rollout_size

    def forward(self, buffer: RolloutBuffer, next_state, done, agent, device, _buffer_updated):
        if not buffer.reached_rollout_size():
            return

        state_t = torch.as_tensor(next_state, dtype=torch.float32, device=device)
        if state_t.dim() == 1:
            state_t = state_t.unsqueeze(0)
        _, final_values = agent(state_t)
        final_values = final_values.squeeze(-1).cpu().numpy()

        is_vectorized = isinstance(done, (np.ndarray, list)) and len(np.atleast_1d(done)) > 1
        if not is_vectorized:
            done, final_values = [done], [final_values]

        num_envs = len(done)
        for i in range(num_envs):
            boot_val = 0.0 if done[i] else final_values[i]
            buffer.buffer[-(num_envs - i)].bootstrap_value = boot_val


class BootStrappingNodeMujoco(Node):
    def __init__(self, rollout_size):
        super().__init__("Bootstrapping",
                         ["buffer", "next_state", "done", "critic", "device", "_buffer_updated"],
                         [], no_grad=True)
        self.rollout_size = rollout_size

    def forward(self, buffer: RolloutBuffer, next_state, done, critic, device, _buffer_updated):
        if not buffer.reached_rollout_size():
            return

        state_t = torch.as_tensor(next_state, dtype=torch.float32, device=device)
        if state_t.dim() == 1:
            state_t = state_t.unsqueeze(0)
        final_values = critic(state_t)
        final_values = final_values.squeeze(-1).cpu().numpy()

        is_vectorized = isinstance(done, (np.ndarray, list)) and len(np.atleast_1d(done)) > 1
        if not is_vectorized:
            done, final_values = [done], [final_values]

        num_envs = len(done)
        for i in range(num_envs):
            boot_val = 0.0 if done[i] else final_values[i]
            buffer.buffer[-(num_envs - i)].bootstrap_value = boot_val


class KUpdateNode(Node):
    def __init__(self, name, inputs, outputs, inner_graph, iterations, batch_size):
        super().__init__(name, inputs, outputs)
        self.inner_graph = inner_graph
        self.iterations = iterations
        self.batch_size = batch_size

    def forward(self, dataset_dict, inner_context):
        dataset_size = len(next(iter(dataset_dict.values())))
        indices = np.arange(dataset_size)
        p_losses, v_losses, entropies = [], [], []

        for _ in range(self.iterations):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, self.batch_size):
                idx = indices[start: start + self.batch_size]

                current_inner_context = inner_context.copy()

                minibatch_dict = {key: tensor[idx] for key, tensor in dataset_dict.items()}

                current_inner_context.update({
                    "minibatch_dict": minibatch_dict,
                    "minibatch_fields": ["states", "actions", "logps", "advantages", "returns"]
                })

                self.inner_graph.run(current_inner_context)

                p_losses.append(current_inner_context["policy_loss"].item())
                v_losses.append(current_inner_context["value_loss"].item())
                entropies.append(current_inner_context["entropy"].item())

        return {
            "losses/policy_loss": np.mean(p_losses),
            "losses/value_loss": np.mean(v_losses),
            "losses/entropy": np.mean(entropies)
        }


class RepeatNode(Node):
    def __init__(self, name, inputs, outputs, inner_graph, iterations):
        super().__init__(name, inputs, outputs)
        self.inner_graph = inner_graph
        self.iterations = iterations

    def forward(self, loop_context):
        current_context = loop_context.copy()

        for i in range(self.iterations):
            current_context["_iteration"] = i
            self.inner_graph.run(current_context)

        return current_context


class ConditionNode(Node):
    def __init__(self, name, condition_key, func_1, func_2, inputs, outputs, no_grad=True):
        self.condition_key = condition_key
        self.function_1 = func_1
        self.function_2 = func_2
        self.no_grad = no_grad

        super().__init__(name, [condition_key] + inputs, outputs)

    def forward(self, condition, *args):
        import torch
        if self.no_grad:
            with torch.no_grad():
                if condition:
                    return self.function_1(*args)
                return self.function_2(*args)
        else:
            if condition:
                return self.function_2(*args)
            return self.function_1(*args)


class TimerNode(Node):
    def __init__(self, name, timer_inputs, data_inputs, outputs):
        super().__init__(name, timer_inputs + data_inputs, outputs)
        self.num_timer_args = len(timer_inputs)

    def forward(self, *args):
        step = args[0]
        freq = args[1]

        #Payload
        data = args[self.num_timer_args:]

        if step % freq == 0:
            return data[0] if len(data) == 1 else tuple(data)
        else:
            if len(self.outputs) == 1:
                return Signal.NOSIGNAL
            return tuple([Signal.NOSIGNAL] * len(self.outputs))