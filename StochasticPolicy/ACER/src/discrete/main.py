
import random
import numpy as np
import torch

from backend.StochasticPolicy.ACER.src.discrete.ACERDataCollector import ACERDataCollector
from backend.StochasticPolicy.ACER.src.discrete.ACERTrainProcessor import ACERTrainProcessor
from backend.StochasticPolicy.ACER.src.discrete.ACERTrainer import ACERTrainer
from backend.Utils.src.BatchTransitioner import TransitionSpec, TransitionFactory
from backend.Utils.src.EnvFactory import AtariEnvFactory
from backend.Utils.src.EnviromentHandler import EnvironmentHandler
from backend.Utils.src.ReplayBuffer import ReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def acer_evaluate(trainer, env_factory, episodes=100):
    scores = []
    for _ in range(episodes):
        env = env_factory.create()
        state, _ = env.reset()
        state = np.asarray(state, dtype=np.float32)
        noops = random.randint(0, 30)
        for _ in range(noops):
            state, _, terminated, truncated, _ = env.step(0)
            done = terminated or truncated
            state = np.asarray(state, dtype=np.float32)
            if done:
                state, _ = env.reset()
                state = np.asarray(state, dtype=np.float32)

        done = False
        episode_reward = 0.0

        while not done:
            state_t = torch.from_numpy(state).unsqueeze(0).float().to(device)
            logits = trainer.actor(state_t)
            action = torch.argmax(logits, dim=-1).item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            state = np.asarray(next_state, dtype=np.float32)
            episode_reward += reward

        scores.append(episode_reward)

    return np.mean(scores)

def main():
    env_name = "ALE/Breakout-v5"
    seed = 100
    max_timesteps = 100000
    num_envs = 16
    batch_size = 32
    learning_rate = 3e-4
    hidden_dim = 200
    tau = 1.0
    buffer_size = int(1e6)
    seq_len = 20
    replay_ratio = 4
    trust_region_delta = 1.0
    gamma = 0.99
    reward_scale = 1.0

    factory = AtariEnvFactory(env_name)

    env_handlers = [EnvironmentHandler(factory, seed + i, reward_scale=reward_scale) for i in range(num_envs)]
    # Transition spec for ACER
    spec = TransitionSpec(["state", "action", "reward", "next_state", "mask", "mu_logp", "mu_logits"])
    transition_factory = TransitionFactory(spec)

    # Trainer
    trainer = ACERTrainer(
        state_size=env_handlers[0].state_dim,
        action_size=env_handlers[0].action_dim,
        hidden_size=hidden_dim,
        learning_rate=learning_rate,
        gamma=gamma,
        tau=tau,
        trust_region_delta=trust_region_delta
    )

    buffer = ReplayBuffer(spec, buffer_size, batch_size)

    collectors = [ACERDataCollector(trainer, env_handlers[i], buffer, transition_factory, device, seq_len)
                  for i in range(num_envs)]
    train_process = ACERTrainProcessor(trainer, buffer, seq_len, replay_ratio, batch_size, tau)
    # sync_process = SyncProcessor(trainer.actor, trainer.trust_region_actor, tau, sync_freq=1)

    for step in range(max_timesteps):
        on_policy_rollouts = []
        for collector in collectors:
            rollout = collector.run()
            if rollout is not None:
                on_policy_rollouts.append(rollout)

        if len(on_policy_rollouts) == num_envs:
            train_process.run(on_policy_rollouts)

        if step % 10000 == 0 and step > 0:
            score = acer_evaluate(trainer, factory, episodes=5)
            print(f"[EVAL] Step {step}: Mean Score = {score}")


if __name__ == "__main__":
    main()
