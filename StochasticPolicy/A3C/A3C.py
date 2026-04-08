import torch
import torch.multiprocessing as mp
from torch import optim

from backend.StochasticPolicy.A3C.src.A3CDataCollectionProcessor import A3CDataCollectionProcessor
from backend.StochasticPolicy.A3C.src.A3CTrainingProcessor import A3CTrainingProcessor
from backend.CommonModels.src.ActorCriticA3C import ActorCritic
from backend.Utils.src.BatchTransitioner import TransitionSpec, TransitionFactory
from backend.Utils.src.EnvFactory import GymEnvFactory
from backend.Utils.src.EnviromentHandler import EnvironmentHandler
from backend.Utils.src.SyncProcessor import SyncProcessor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class A3CWorker:
    def __init__(self, worker_id, env_name, seed, t_max, gamma, global_net, optimizer, counter, factory, obs_dim,
                 act_dim, hidden):
        torch.manual_seed(seed + worker_id)

        # Local network (theta′)
        self.local_net = ActorCritic(obs_dim, act_dim, hidden).to(device)
        self.local_net.load_state_dict(global_net.state_dict())
        self.global_net = global_net
        self.optimizer = optimizer
        self.counter = counter

        self.env = EnvironmentHandler(env_name, seed + worker_id)

        self.collector = A3CDataCollectionProcessor(
            self.local_net, self.env, t_max, factory, gamma
        )

        self.trainer = A3CTrainingProcessor(
            self.global_net, self.local_net, self.optimizer,
            gamma, entropy_coef=0.001, max_grad_norm=40.0
        )

        self.syncer = SyncProcessor(self.global_net, self.local_net, tau=1.0, sync_freq=1)

    def run(self):
        while not self.counter.reached_limit():
            self.syncer.run()
            rollout = self.collector.run()
            steps = len(rollout)

            self.counter.increment(steps)

            self.trainer.run(rollout)


class GlobalCounter:
    def __init__(self, T, T_max):
        self.T = T
        self.T_max = T_max

    def increment(self, n):
        with self.T.get_lock():
            self.T.value += n
            return self.T.value

    def reached_limit(self):
        with self.T.get_lock():
            return self.T.value >= self.T_max


def worker_entry(worker_id, env_name, seed, t_max, gamma, global_net, optimizer, T, T_max, obs_dim, act_dim, hidden,
                 spec_fields):
    counter = GlobalCounter(T, T_max)
    spec = TransitionSpec(spec_fields)
    factory = TransitionFactory(spec)

    worker = A3CWorker(worker_id, env_name, seed, t_max, gamma, global_net, optimizer, counter, factory, obs_dim,
                       act_dim, hidden)

    worker.run()


def main():
    env_name = "CartPole-v1"
    num_workers = 4
    t_max = 20
    gamma = 0.99
    lr = 7e-4
    hidden_size = 64
    seed = 42
    T_max = 1_000_000

    torch.manual_seed(seed)
    gym_factory = GymEnvFactory(env_name)
    env = EnvironmentHandler(gym_factory, seed)
    obs_dim, act_dim, _ = env.get_env_specs()
    global_net = ActorCritic(obs_dim, act_dim, hidden_size).to(device)
    global_net.share_memory()
    optimizer = optim.RMSprop(global_net.parameters(), lr=lr)
    T = mp.Value('i', 0)
    spec_fields = ["state", "action", "reward", "value", "log_prob", "done", "entropy"]
    processes = []
    for worker_id in range(num_workers):
        p = mp.Process(
            target=worker_entry,
            args=(worker_id, env_name, seed, t_max, gamma, global_net, optimizer, T, T_max, obs_dim, act_dim,
                  hidden_size, spec_fields)
        )
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    print("Training finished.")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
