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
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStack

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def acer_evaluate(trainer, env_factory, episodes=100):
    scores = []

    for _ in range(episodes):
        env = env_factory.make_env()  # fresh evaluation env
        state = env.reset()
        state = np.array(state, dtype=np.float32)
        noops = random.randint(0, 30)
        for _ in range(noops):
            state, _, done, _ = env.step(0)
            state = np.array(state, dtype=np.float32)
            if done:
                state = env.reset()
                state = np.array(state, dtype=np.float32)

        done = False
        episode_reward = 0.0

        while not done:
            state_t = torch.from_numpy(state).unsqueeze(0).float().to(device)
            logits = trainer.actor(state_t)
            action = torch.argmax(logits, dim=-1).item()  # deterministic π

            next_state, reward, done, _ = env.step(action)
            next_state = np.array(next_state, dtype=np.float32)

            episode_reward += reward  # raw reward
            state = next_state

        scores.append(episode_reward)

    return np.mean(scores)


def main():
    env_name = "ALE/Pong-v5"
    seed = 100
    max_timesteps = 1000000
    batch_size = 32
    learning_rate = 3e-4
    hidden_dim = 200
    tau = 0.01
    buffer_size = int(1e6)
    seq_len = 20
    replay_ratio = 4
    trust_region_delta = 0.1
    gamma = 0.99
    reward_scale = 1.0

    factory = AtariEnvFactory(env_name)
    env_handler = EnvironmentHandler(factory, seed, reward_scale=reward_scale)

    # Transition spec for ACER
    spec = TransitionSpec(["state", "action", "reward", "next_state", "mask", "mu_logp", "mu_logits"])
    factory = TransitionFactory(spec)

    # Trainer
    trainer = ACERTrainer(
        state_size=env_handler.state_dim,
        action_size=env_handler.action_dim,
        hidden_size=hidden_dim,
        learning_rate=learning_rate,
        gamma=gamma,
        tau=tau,
        trust_region_delta=trust_region_delta
    )

    buffer = ReplayBuffer(spec, buffer_size, batch_size)

    # Processors
    collector = ACERDataCollector(trainer, env_handler, buffer, factory, device, seq_len)
    train_process = ACERTrainProcessor(trainer, buffer, seq_len, replay_ratio, batch_size, tau)
    # sync_process = SyncProcessor(trainer.actor, trainer.trust_region_actor, tau, sync_freq=1)

    # Main loop
    for step in range(max_timesteps):
        collector.run()
        train_process.run()
        # sync_process.run() #TODO: This still gets handled on a lower level because of the on Policy off Policy rhythm
        if step % 100000 == 0 and step > 0:
            eval_score = acer_training_eval(trainer, factory, episodes=10)
            print(f"[EVAL] Step {step}: Mean Score = {eval_score}")


if __name__ == "__main__":
    main()
