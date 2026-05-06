import time
from copy import deepcopy

import torch
import wandb

from backend.CommonModels.src.RainbowDuellingDQN import RainbowDuellingDQN
from TobeTranslatedAlgorithms.RainbowDQN.src.ActionHandler import GreedyPolicy
from TobeTranslatedAlgorithms.RainbowDQN.src.DataCollectionProcessor import DataCollectionProcessor
from TobeTranslatedAlgorithms.RainbowDQN.src.TrainProcessor import TrainProcessor
from backend.Utils.src.BatchTransitioner import TransitionSpec, TransitionFactory
from backend.Utils.src.EnvFactory import GymEnvFactory
from backend.Utils.src.EnviromentHandler import EnvironmentHandler
from backend.Utils.src.PrioReplayBuffer import PrioReplayBuffer
from backend.Utils.src.StepBuffer import StepBuffer
from backend.Utils.src.SyncProcessor import SyncProcessor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    start_time = time.time()
    env_name = "CartPole-v1"
    sync_freq = 40
    hidden_size = 128
    batch_size = 64
    atoms_size = 51
    max_buffer_size = 1000000
    tau = 1.0
    gamma = 0.99
    max_steps = 100000
    seed = 42
    lr = 5e-4
    max_grad_norm = 10.0
    v_min: int = 0  # dependent on used environment, so be careful
    v_max: int = 500

    wandb.init(
        entity="michael_dohmen-",
        project="my-RainbowDQN-benchmarks",
        config={
            "env_id": env_name,
            "exp_name": "Rainbow-CartPole-v1",
            "seed": seed,
            "max_buffer_size": max_buffer_size,
            "batch_size": batch_size,
            "max_steps": max_steps,
            "lr": lr,
            "gamma": gamma,
            "sync_freq": sync_freq,
            "tau": tau,
            "max_grad_norm": max_grad_norm,
            "v_min": v_min,
            "v_max": v_max,
            "atoms_size": atoms_size,
        }
    )

    spec = TransitionSpec(["state", "action", "reward", "next_state", "done"])
    factory = TransitionFactory(spec)
    gym_factory = GymEnvFactory(env_name)
    env = EnvironmentHandler(gym_factory, seed)
    obs_size, action_size, _ = env.get_env_specs()

    behavior_net = RainbowDuellingDQN(obs_size, hidden_size, action_size, atoms_size).to(device)
    target_net = deepcopy(behavior_net).to(device)

    optimizer = torch.optim.Adam(behavior_net.parameters(), lr)

    step_buffer = StepBuffer(factory, lookahead_n=3, gamma=gamma)
    buffer = PrioReplayBuffer(spec, max_buffer_size, batch_size)

    collector = DataCollectionProcessor(behavior_net, env, buffer, step_buffer, GreedyPolicy(), factory,
                                        device, v_min, v_max, atoms_size)

    train_process = TrainProcessor(buffer, behavior_net, target_net, optimizer, gamma, device, v_min, v_max, atoms_size,
                                   max_grad_norm)

    sync_process = SyncProcessor(behavior_net, target_net, tau, sync_freq)

    for step in range(max_steps):
        collector.run()
        metrics = train_process.run()
        sync_process.run()

        if metrics and step % 400 == 0:
            metrics["charts/SPS"] = int(step / (time.time() - start_time))
            wandb.log(metrics, step=step)
        #if step % 1_000 == 0 and step > 0:
            #avg_score = evaluate_policy(behavior_net, eval_env, episodes=10, device=device)
         #   wandb.log({"charts/eval_avg_score": avg_score}, step=step)


if __name__ == "__main__":
    main()
