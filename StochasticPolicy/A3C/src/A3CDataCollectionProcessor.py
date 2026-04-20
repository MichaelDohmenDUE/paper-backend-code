import torch
from torch.distributions import Categorical

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class A3CDataCollectionProcessor:
    def __init__(self, local_net, env, t_max, factory, gamma):
        self.local_net = local_net  # θ′, θ′_v
        self.env = env
        self.t_max = t_max
        self.factory = factory
        self.gamma = gamma

        self.episode_reward = 0.0
        self.episode_count = 0
        self.avg_reward = 0.0
        self.beta = 0.99

        self.episode_timesteps = 0

        self.reset_episode()

    def reset_episode(self):
        self.state = self.env.reset()
        self.done = False
        self.episode_reward = 0.0
        self.episode_timesteps = 0

    def run(self):
        rollout = []
        t = 0

        while t < self.t_max and not self.done:
            state_t = torch.tensor(self.state, dtype=torch.float32, device=device).unsqueeze(0)

            logits, value = self.local_net(state_t)
            probs = torch.softmax(logits, dim=-1)
            dist = Categorical(probs)
            action = dist.sample()

            next_state, reward, done, _ = self.env.step(action.item())
            self.episode_reward += reward

            transition = self.factory.forward(state=state_t, action=action, reward=float(reward),
                                              value=value.squeeze(-1), log_prob=dist.log_prob(action), done=bool(done),
                                              entropy=dist.entropy())

            rollout.append(transition)

            self.state = next_state
            self.done = done
            t += 1

        if self.done:
            self.avg_reward = self.beta * self.avg_reward + (1 - self.beta) * self.episode_reward
            print(f"[Worker] Episode {self.episode_count} Reward: {self.episode_reward:.2f}  MA: {self.avg_reward:.2f}")
            self.episode_count += 1
            self.reset_episode()

        return rollout
