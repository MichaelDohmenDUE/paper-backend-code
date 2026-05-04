import time

import torch
import wandb
from torch import optim

from ReplayBuffer import ReplayBuffer
from backend.CommonModels.src.CriticPPO import CriticPPO
from backend.CommonModels.src.DiscreteActorPPO import DiscreteActorPPO
from backend.StochasticPolicy.PPO.discrete.src.DataCollectorGraphMujoco import DataCollectionProcessor
from backend.StochasticPolicy.PPO.discrete.src.PPOTrainer import PPOTrainerProcessor
from backend.Utils.src.BatchTransitioner import TransitionFactory
from backend.Utils.src.BatchTransitioner import TransitionSpec
from backend.Utils.src.EnvFactory import GymEnvFactory
from backend.Utils.src.EnviromentHandler import VecEnvironmentHandler
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
    eval_episodes = 10
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
        project="my-ppo-benchmarks",
        group=algo_name,
        name=f"{algo_name}-{env_name}-seed-{seed}",
        tags=[env_name, "baseline", algo_name],
        reinit=True,
        entity="michael_dohmen-",
        config={
            "env_id": env_name,
            "exp_name": f"PPO_{env_name}_seed-{seed}",
            "seed": seed,
            "rollout_size": rollout_size,
            "batch_size": batch_size,
            "epochs": epochs,
            "lr": lr,
            "gamma": gamma,
            "lam": lam,
            "max_steps": max_steps,
            "offset": offset,
            "eval_freq": eval_freq,
            "eval_episodes": eval_episodes,
            "device": device
        }
    )

    spec = TransitionSpec(["state", "action", "logp", "reward", "done", "value", "bootstrap_value"])
    replay_spec = TransitionSpec(["state", "action", "logp", "advantage", "return"])
    transition_factory = TransitionFactory(spec)
    factory = GymEnvFactory(env_name)
    env_handler = VecEnvironmentHandler(factory, seed, num_envs=num_envs)
    eval_env_handler = VecEnvironmentHandler(factory, seed + offset, num_envs=1)
    state_dim, action_dim, _ = env_handler.get_env_specs()
    state_dim = state_dim[0]
    actor = DiscreteActorPPO(state_dim, action_dim, hidden_dim).to(device)
    critic = CriticPPO(state_dim, hidden_dim).to(device)
    optimizer = optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=lr)

    rollout_buffer = RolloutBuffer(spec, rollout_size=rollout_size)
    replay_buffer = ReplayBuffer(replay_spec, rollout_size, batch_size)

    trainer = PPOTrainerProcessor(actor, critic, optimizer, rollout_buffer, replay_buffer, batch_size, epochs,
                                  gamma=gamma, lam=lam)

    data_collector = DataCollectionProcessor(env_handler, transition_factory, rollout_buffer, rollout_size,
                                             actor, critic, device)
    step = 0
    eval_step = 0
    while step < max_steps:
        metrics = {}
        metrics_ep = data_collector.run()
        if metrics_ep:
            metrics.update(metrics_ep)
        metrics_train = trainer.run()
        if metrics_train:
            metrics.update(metrics_train)
        step = data_collector.context["total_steps"]
        metrics["global_step"] = step

        elapsed_time = time.time() - start_time
        if elapsed_time > 0:
            sps = int(step / elapsed_time)
            metrics["charts/SPS"] = sps
        if eval_step <= step:
            avg_eval_reward = eval_trainer(trainer, eval_env_handler, eval_episodes=eval_episodes)
            metrics["eval/avg_reward"] = avg_eval_reward
            eval_step += eval_freq
        if len(metrics) > 1:
            wandb.log(metrics, step=step)
    # Final Eval
    metrics = {"global_step": step,
               "charts/eval_avg_score": eval_trainer(trainer, eval_env_handler, eval_episodes=eval_episodes)}
    wandb.log(metrics, step=step)
    wandb.finish()


if __name__ == "__main__":
    seed = [0, 1, 2]
    for seed in seed:
        main(seed)
