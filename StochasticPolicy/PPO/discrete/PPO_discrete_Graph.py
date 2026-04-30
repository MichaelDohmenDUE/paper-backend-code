import time

import torch
import wandb
from torch import optim

from backend.CommonModels.src.CriticPPO import CriticPPO
from backend.CommonModels.src.DiscreteActorPPO import DiscreteActorPPO
from backend.StochasticPolicy.PPO.discrete.src.DataCollectorGraphMujoco import DataCollectionProcessor
from backend.StochasticPolicy.PPO.discrete.src.DiscreteActionHandler import ActionHandler
from backend.StochasticPolicy.PPO.discrete.src.PPOTrainerGraph import PPOTrainerProcessor
from backend.Utils.src.BatchTransitioner import TransitionFactory
from backend.Utils.src.BatchTransitioner import TransitionSpec
from backend.Utils.src.EnvFactory import GymEnvFactory
from backend.Utils.src.EnviromentHandler import EnvironmentHandler, VecEnvironmentHandler
from backend.Utils.src.RolloutBuffer import RolloutBuffer
from backend.Utils.src.utils import setting_global_seed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def eval_trainer(trainer, env_handler, eval_episodes=5):
    avg_reward = 0.0
    for _ in range(eval_episodes):
        state = env_handler.reset()
        done = False
        while not done:
            state_t = torch.as_tensor(state, dtype=torch.float32, device=device)
            with torch.no_grad():
                dist = trainer.actor(state_t)
                action = dist.probs.argmax(dim=-1).cpu().numpy()
            next_state, reward, done, _ = env_handler.step(action)
            avg_reward += reward[0]
            state = next_state
            done = done[0]

    avg_reward /= eval_episodes
    print(f"Average Reward over {eval_episodes} episodes: {avg_reward:.3f}")
    return avg_reward


def main(seed):
    start_time = time.time()
    env_name = "CartPole-v1"
    num_envs = 8
    seed = seed
    rollout_size = 2048
    batch_size = 256
    epochs = 4
    max_steps = 1_000_000
    lr = 1e-4
    hidden_dim = 64
    gamma = 0.99
    lam = 0.95
    eval_freq = 50_000
    offset = 100
    setting_global_seed(seed)
    algo_name = "ppo_discrete_mujoco"

    wandb.init(
        entity="michael_dohmen-",
        project="my-ppo-benchmarks",
        name=f"{algo_name}-seed-{seed}",
        tags=[env_name, "baseline_study", algo_name],
        config={
            "env_id": env_name,
            "exp_name": "my_ppo_cartpole",
            "seed": seed,
            "rollout_size": rollout_size,
            "batch_size": batch_size,
            "epochs": epochs,
            "lr": lr,
            "gamma": gamma,
            "lam": lam,
            "max_steps": max_steps,
            "offset": offset,
        }
    )

    spec = TransitionSpec(["state", "action", "logp", "reward", "done", "value", "bootstrap_value"])
    transition_factory = TransitionFactory(spec)
    factory = GymEnvFactory(env_name)
    env_handler = VecEnvironmentHandler(factory, seed, num_envs=num_envs)
    eval_env_handler = VecEnvironmentHandler(factory, seed + offset, num_envs=1)
    state_dim, action_dim, _ = env_handler.get_env_specs()
    state_dim = state_dim[0]
    actor = DiscreteActorPPO(state_dim, action_dim, hidden_dim).to(device)
    critic = CriticPPO(state_dim, hidden_dim).to(device)
    optimizer = optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=lr)

    rollout_buffer = RolloutBuffer(spec, rollout_size= rollout_size)

    trainer = PPOTrainerProcessor(actor, critic, optimizer, rollout_buffer, batch_size, epochs, gamma=gamma, lam=lam)

    data_collector = DataCollectionProcessor(env_handler, transition_factory, rollout_buffer, rollout_size,
                                             actor, critic, device)
    steps = 0
    eval_step = 0
    while steps < max_steps:
        metrics_ep = data_collector.run()
        metrics_train = trainer.run()
        all_metrics = {**metrics_ep, **metrics_train, "charts/SPS": int(steps / (time.time() - start_time)),
                       "global_step": data_collector.context["total_steps"]}
        steps = data_collector.context["total_steps"]

        if eval_step <= steps:
            avg_eval_reward = eval_trainer(trainer, eval_env_handler, eval_episodes=10)
            all_metrics["eval/avg_reward"] = avg_eval_reward
            eval_step += eval_freq
        wandb.log(all_metrics, step=steps)
    wandb.finish()

if __name__ == "__main__":
    seeds = [0, 1, 2]
    for seed in seeds:
        main(seed)
