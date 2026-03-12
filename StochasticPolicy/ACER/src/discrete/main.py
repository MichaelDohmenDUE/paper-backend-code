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

def main():
    env_name = "ALE/SpaceInvaders-v5"
    seed = 100
    max_timesteps = 1000000
    batch_size = 32
    learning_rate = 3e-4
    hidden_dim = 200
    tau = 0.01
    buffer_size = int(1e6)
    seq_len = 10
    replay_ratio = 2
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
    collector = ACERDataCollector(trainer, env_handler, buffer, factory, device)
    train_process = ACERTrainProcessor(trainer, buffer, seq_len, replay_ratio, batch_size, tau)
    # sync_process = SyncProcessor(trainer.actor, trainer.trust_region_actor, tau, sync_freq=1)

    # Main loop
    for step in range(max_timesteps):
        collector.run()
        train_process.run()
        # sync_process.run() #TODO: This still gets handled on a lower level because of the on Policy off Policy rhythm


if __name__ == "__main__":
    main()
