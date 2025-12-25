from copy import deepcopy
import torch
from torch import nn
import random
import torch.nn.functional as F


from backend.Utils.src.BatchTransitioner import TransitionSpec, TransitionFactory
from backend.Utils.src.EnviromentHandler import EnvironmentHandler
from backend.Utils.src.ReplayBuffer import ReplayBuffer
from backend.Utils.src.SyncProcessor import SyncProcessor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EpsilonGreedyPolicy:
    def __init__(self, epsilon: float):
        self.epsilon = epsilon

    def select_action(self, q_values: torch.Tensor) -> torch.Tensor:
        """ Epsilon-greedy policy, returns random action if random number is < epsilon, else greedy action
                Randomly samples from max actions if there is a tie.
             """
        actions = torch.arange(len(q_values)).to(q_values.device)
        max_q_value = torch.max(q_values).to(q_values.device)
        max_idx = (q_values == max_q_value).to(torch.int64)
        greedy_action = random.choice(actions[max_idx == 1])

        return random.choice(actions) if random.random() < self.epsilon else greedy_action


class DataCollectionProcessor:
    def __init__(self, policy: nn.Module, env: EnvironmentHandler, buffer: ReplayBuffer, action_selector: EpsilonGreedyPolicy, transition_factory: TransitionFactory, device: torch.device):
        self.policy = policy
        self.env = env
        self.buffer = buffer
        self.state = env.reset()
        self.done = False
        self.action_selector = action_selector
        self.transition_factory = transition_factory
        self.device = device
        # Logging
        self.episode_count = 0
        self.episode_reward = 0
        self.total_steps = 0

    def run(self) -> None:
        if self.done:
            if self.episode_count % 10 == 0:
                print(f"Episode [{self.episode_count}] {self.episode_reward}")

            self.episode_count += 1
            self.episode_reward = 0.0

            self.state = self.env.reset()
            self.done = False
        with torch.no_grad():
            state_tensor = torch.as_tensor(self.state, dtype=torch.float32, device=self.device)
            q_values = self.policy(state_tensor)
        action = self.action_selector.select_action(q_values=q_values)
        next_state, reward, done, done_bool = self.env.step(action.item(), self.total_steps)
        self.done = done

        transition = self.transition_factory.create( state=self.state, action=action.item(), reward=reward, next_state=next_state, done=self.done )
        self.buffer.append(transition)
        self.episode_reward += reward

        self.state = next_state

        self.total_steps += 1

class TrainProcessor:
    def __init__(self, buffer: ReplayBuffer, behavior_net: nn.Module, target_net: nn.Module,
                 optimizer: torch.optim.Optimizer, gamma: float,  device: torch.device):
        self.buffer = buffer
        self.behavior_net = behavior_net.to(device)
        self.target_net = target_net.to(device)
        self.optimizer = optimizer
        self.gamma = gamma
        self.device = device

    def run(self):
        if len(self.buffer) < self.buffer.batch_size:
            return

        batch = self.buffer.sample_batch()

        states_tensor      = batch["state"].to(self.device)
        actions_tensor     = batch["action"].to(self.device)
        rewards_tensor     = batch["reward"].to(self.device)
        next_states_tensor = batch["next_state"].to(self.device)
        dones_tensor       = batch["done"].to(self.device)


        qsa_behavior = self.behavior_net(states_tensor).gather(1, actions_tensor)  # ^y

        qs_target = self.target_net(next_states_tensor)  # batch_size x action_dim
        qsa_target = torch.max(qs_target, dim=1).values.unsqueeze(-1).detach()
        target = rewards_tensor + self.gamma * qsa_target * (1.0 - dones_tensor)
        target = target.detach()

        # ToDo: How to model dependent steps without forwarding anything
        self.optimizer.zero_grad()
        loss = F.mse_loss(qsa_behavior, target)
        loss.backward()
        self.optimizer.step()


def main():
    # initialization
    # ToDo: Constants as global variables or member variables
    lr = 1e-3
    epsilon = 0.2
    env_name = "CartPole-v1"
    sync_freq = 40
    hidden_size = 32
    batch_size = 64
    max_buffer_size = 10000
    tau = 1.0
    gamma = 0.99
    max_steps = 100000
    spec = TransitionSpec(["state", "action", "reward", "next_state", "done"])
    factory = TransitionFactory(spec)
    seed = 42

    env = EnvironmentHandler(env_name, seed)
    obs_size, action_size, max_action = env.get_env_specs()

    behavior_net = nn.Sequential(nn.Linear(obs_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, action_size)).to(device)
    optimizer = torch.optim.Adam(behavior_net.parameters(), lr)

    target_net = deepcopy(behavior_net).to(device)

    buffer = ReplayBuffer(spec, max_buffer_size, batch_size)
    collector = DataCollectionProcessor(behavior_net, env, buffer, EpsilonGreedyPolicy(epsilon), factory, device)
    train_process = TrainProcessor(buffer, behavior_net, target_net, optimizer, gamma, device)
    sync_process = SyncProcessor(behavior_net, target_net, tau, sync_freq)


    for step in range(max_steps):
        collector.run()
        train_process.run()
        sync_process.run()


if __name__ == '__main__':
    main()
