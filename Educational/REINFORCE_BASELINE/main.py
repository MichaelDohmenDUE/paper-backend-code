import time

import wandb

from backend.Utils.src import RolloutBuffer
from backend.Utils.src.BatchTransitioner import TransitionFactory, TransitionSpec
from backend.Utils.src.EnvFactory import GymEnvFactory
from backend.Utils.src.RolloutBuffer import RolloutBuffer
import torch

from backend.CommonModels.src.Policy_Reinforce_Baseline import PolicyReinforceBaseline
from backend.Utils.src.EnviromentHandler import VecEnvironmentHandler
from Educational.REINFORCE_BASELINE.src.DataCollector import DataCollectionProcessor
from Educational.REINFORCE_BASELINE.src.ReinforceTrainer import REINFORCETrainer
from backend.Utils.src.utils import setting_global_seed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_policy(policy, env_handler, device, episodes=10):
    policy.eval()
    total_reward = 0
    for _ in range(episodes):
        state = env_handler.reset()
        done = False
        while not done:
            state_t = torch.as_tensor(state, dtype=torch.float32, device=device)
            with torch.no_grad():
                action_logits, _ = policy(state_t)
                action = [torch.argmax(action_logits, dim=1).item()]
            next_state, reward, done, _ = env_handler.step(action)
            total_reward += reward
            state = next_state
    env_handler.reset()
    policy.train()
    return total_reward / episodes



def main(seed):
    """
    Related Paper: https://link.springer.com/article/10.1007/BF00992696
    """
    start_time = time.time()
    learn_rate = 1e-4
    max_steps = 200_000
    hidden_dim = 64
    env_name = "CartPole-v1"
    beta = 0.01
    gamma = 0.99
    eval_freq = 1000
    eval_episodes = 10
    algo_name = "REINFORCE_WITH_BASELINE"
    opt = "Adam"
    setting_global_seed(seed)

    wandb.init(
        entity="michael_dohmen-",
        project="Educational_Benchmarks",
        group=algo_name,
        name=f"{algo_name}-seed-{seed}",
        tags=[env_name, "study"],
        reinit=True,
        config={
            "env_id": env_name,
            "exp_name": "REINFORCEBASELINE-CartPole-v1",
            "seed": seed,
            "lr": learn_rate,
            "gamma": gamma,
            "max_steps": max_steps,
            "hidden_dim": hidden_dim,
            "eval_freq": eval_freq,
            "eval_episodes": eval_episodes,
            "algo_name": algo_name,
            "optimizer": opt,
        }
    )

    spec = TransitionSpec(["state", "logp", "reward", "done"])
    transition_factory = TransitionFactory(spec)
    replay_buffer = RolloutBuffer(spec, rollout_size=500)

    gym_factory = GymEnvFactory(env_name)
    env_handler = VecEnvironmentHandler(gym_factory, seed=seed, num_envs=1)
    eval_env_handler = VecEnvironmentHandler(gym_factory, seed=seed + 100, num_envs=1)

    observation_size, action_size, _ = env_handler.get_env_specs()
    observation_size = observation_size[0]
    policy = PolicyReinforceBaseline(observation_size, action_size, hidden_dim=hidden_dim).to(device)

    optimizer = torch.optim.Adam(policy.parameters(), lr=learn_rate)

    data_collector = DataCollectionProcessor(env_handler, transition_factory, replay_buffer, policy, device)

    trainer = REINFORCETrainer(replay_buffer, policy, optimizer, beta, gamma, device=device)
    step = 0
    eval_step = 0
    while step < max_steps:
        metrics_ep = data_collector.run()
        metrics_train = trainer.run()
        step = data_collector.total_steps
        all_metrics = {**metrics_ep, **metrics_train, "charts/SPS": int(step / (time.time() - start_time)),
                       "global_step": data_collector.total_steps}

        if step >= eval_step:
            avg_eval_reward = evaluate_policy(policy, eval_env_handler, device, eval_episodes)
            all_metrics["eval/avg_reward"] = avg_eval_reward
            eval_step += eval_freq
        wandb.log(all_metrics, step=step)
    wandb.finish()


if __name__ == "__main__":
    for seed in range(10):
        main(seed)
