import torch

from TobeTranslatedAlgorithms.A3C.src.A3CNodes import build_a3c_rollout_graph
from backend.Utils.src.NodeLib.Node import Graph

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class A3CDataCollectionProcessor:
    def __init__(self, local_net, env, t_max, factory, gamma):
        self.local_net = local_net
        self.env = env
        self.t_max = t_max
        self.factory = factory
        self.gamma = gamma

        self.beta = 0.99
        self.state = self.env.reset()
        self.done = False
        self.avg_reward = 0.0
        self.episode_count = 0
        self.episode_reward = 0.0
        self.episode_timesteps = 0

        self.graph = Graph(build_a3c_rollout_graph())

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
            ctx = {
                "state": self.state,
                "device": device,
                "local_net": self.local_net,
                "env": self.env,
                "factory": self.factory,
                "episode_timesteps": self.episode_timesteps,

                "rollout": rollout,
                "episode_reward": self.episode_reward,
                "avg_reward": self.avg_reward,
                "episode_count": self.episode_count,
                "beta": self.beta,
                "t": t,
                "done": self.done,
            }

            self.graph.run(ctx)

            self.state = ctx["state"]
            self.done = ctx["done"]
            self.episode_reward = ctx["episode_reward"]
            self.avg_reward = ctx["avg_reward"]
            self.episode_count = ctx["episode_count"]
            t = ctx["t"]

        if self.done:
            print(f"[Worker] Episode {self.episode_count} Reward: {self.episode_reward:.2f}  MA: {self.avg_reward:.2f}")
            self.reset_episode()

        return rollout
