import time

import numpy as np

import wandb

from backend.Utils.src import RolloutBuffer
from backend.Utils.src.BatchTransitioner import TransitionFactory, TransitionSpec
from backend.Utils.src.EnvFactory import GymEnvFactory
from backend.Utils.src.RolloutBuffer import RolloutBuffer
import torch

from backend.CommonModels.src.Policy_Reinforce_Baseline import PolicyReinforceBaseline
from backend.Utils.src.EnviromentHandler import EnvironmentHandler
from Educational.ACTOR_CRITIC.src.DataCollector import DataCollectionProcessor
from Educational.ACTOR_CRITIC.src.Trainer import Trainer
from backend.Utils.src.utils import setting_global_seed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(seed):
    """
    Related Paper: https://link.springer.com/article/10.1007/BF00992696
    """
    start_time = time.time()
    learn_rate = 1e-3
    max_steps = 100_000
    seed = seed
    hidden_dim = 64
    env_name = "CartPole-v1"
    beta = 0.01
    gamma = 0.99
    eval_freq = 1000
    eval_episodes = 10
    algo_name = "ACTOR_CRITIC"
    setting_global_seed(seed)

    wandb.init(
        entity="michael_dohmen-",
        project="Educational_Benchmarks",
        group=algo_name,
        name=f"{algo_name}-seed-{seed}",
        tags=[env_name, "baseline_study"],
        reinit=True,
        config={
            "env_id": env_name,
            "exp_name": "ACTORCRITIC-CartPole-v1",
            "seed": seed,
            "lr": learn_rate,
            "gamma": gamma,
            "max_steps": max_steps,
            "hidden_dim": hidden_dim,
            "eval_freq": eval_freq,
            "eval_episodes": eval_episodes,
            "algo_name": algo_name,

        }
    )

    spec = TransitionSpec(["state", "logp", "reward", "done", "next_state"])
    transition_factory = TransitionFactory(spec)
    replay_buffer = RolloutBuffer(spec)

    gym_factory = GymEnvFactory(env_name)
    env_handler = EnvironmentHandler(gym_factory, seed=seed)
    eval_env_handler = EnvironmentHandler(gym_factory, seed=seed + 100)
    observation_size, action_size, _ = env_handler.get_env_specs()
    policy = PolicyReinforceBaseline(observation_size, action_size, hidden_dim=hidden_dim).to(device)

    optimizer = torch.optim.SGD(policy.parameters(), lr=learn_rate)

    data_collector = DataCollectionProcessor(env_handler, transition_factory, replay_buffer, policy, device)

    trainer = Trainer(replay_buffer, policy, optimizer, beta, gamma, device=device)
    step = 0
    eval_step = 0
    while step < max_steps:
        metrics_ep = data_collector.run()
        metrics_train = trainer.run()
        all_metrics = {**metrics_ep, **metrics_train, "charts/SPS": int(step / (time.time() - start_time)),
                       "global_step": data_collector.total_steps}
        step = data_collector.total_steps
        if step > eval_step:
            avg_eval_reward = evaluate_policy(policy, eval_env_handler, device, eval_episodes)
            all_metrics["eval/avg_reward"] = avg_eval_reward
            eval_step += eval_freq
        wandb.log(all_metrics, step=step)
    wandb.finish()
if __name__ == "__main__":
    for seed in range(10):
        main(seed)
