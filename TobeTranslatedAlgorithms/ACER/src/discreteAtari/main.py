import numpy as np
import torch

from TobeTranslatedAlgorithms.ACER.src.discreteAtari.ACERDataCollector import ACERDataCollector
from TobeTranslatedAlgorithms.ACER.src.discreteAtari.ACERTrainProcessor import ACERTrainProcessor
from TobeTranslatedAlgorithms.ACER.src.discreteAtari.ACERTrainer import (ACERTrainer)
from backend.Utils.src.BatchTransitioner import TransitionSpec, TransitionFactory
from backend.Utils.src.EnvFactory import AtariEnvFactory
from backend.Utils.src.EnviromentHandler import EnvironmentHandler, VecEnvironmentHandler
from backend.Utils.src.ReplayBuffer import ReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def acer_evaluate(trainer, vec_env_handler, episodes=10):
    """
    LLMs GEnerated
    """
    scores = []
    for i in range(episodes):
        state = vec_env_handler.reset()
        done = False
        episode_reward = 0.0

        while not done:
            state_t = torch.from_numpy(state).float().to(device)

            with torch.no_grad():
                logits, _ = trainer.model(state_t)
                action = torch.argmax(logits, dim=-1).cpu().numpy()

            next_state, reward, terminated, truncated, _ = vec_env_handler.step_detailed(action)

            done = terminated[0] or truncated[0]
            episode_reward += reward[0]
            state = next_state

        scores.append(episode_reward)
        print(f"Eval Episode {i + 1} Score: {episode_reward}")

    return np.mean(scores)

def main():
    env_name = "PongNoFrameskip-v4"
    seed = 1
    max_timesteps = 10_000_000
    num_envs = 16
    batch_size = 16
    learning_rate = 3e-4
    hidden_dim = 200
    tau = 0.01
    buffer_size = 250_000
    seq_len = 20
    replay_ratio = 4
    trust_region_delta = 0.01
    gamma = 0.99
    warm_up = 20000
    reward_scale = 1.0

    factory = AtariEnvFactory(env_name)

    env_handler = VecEnvironmentHandler(factory, seed, num_envs)
    vec_env_handler = VecEnvironmentHandler(factory, seed + 100, 1)
    state_dim, action_dim, max_action = env_handler.get_env_specs()
    # Transition spec for ACER
    spec = TransitionSpec(["state", "action", "reward", "next_state", "mask", "mu_logp", "mu_logits"])
    transition_factory = TransitionFactory(spec)

    # Trainer
    trainer = ACERTrainer(
        state_size=state_dim,
        action_size=action_dim,
        hidden_size=hidden_dim,
        learning_rate=learning_rate,
        gamma=gamma,
        tau=tau,
        trust_region_delta=trust_region_delta
    )

    buffer = ReplayBuffer(spec, buffer_size, batch_size)

    collector = ACERDataCollector(trainer, env_handler, buffer, transition_factory, device, seq_len)

    train_process = ACERTrainProcessor(trainer, buffer, seq_len, replay_ratio, batch_size, tau)
    # sync_process = SyncProcessor(trainer.actor, trainer.trust_region_actor, tau, sync_freq=1)

    for step in range(max_timesteps):
        on_policy_rollouts = collector.run()
        if on_policy_rollouts is not None and len(buffer) > warm_up:
            #print(f"Buffer Level: {len(buffer)} / {warm_up}")
            train_process.run(on_policy_rollouts)

        if step % 1000 == 0 and step > warm_up :
            score = acer_evaluate(trainer, vec_env_handler, episodes=5)
            print(f"[EVAL] Step {step}: Mean Score = {score}")


if __name__ == "__main__":
    main()
