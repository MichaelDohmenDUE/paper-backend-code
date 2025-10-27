import numpy as np
import torch
import gymnasium as gym

from backend.T3D.src.TD3_Trainer import TD3_Trainer
from backend.Utils.src.ReplayBuffer import ReplayBuffer

def eval_trainer(trainer, env, eval_episodes=5):
    avg_reward = 0
    for _ in range(eval_episodes):
        state, _ = env.reset()
        done = False
        while not done:
            action = trainer.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            avg_reward += reward
            state = next_state
    avg_reward /= eval_episodes
    print(f"Average Reward over {eval_episodes} episodes: {avg_reward:.3f}")
    return avg_reward

def main():
    env_name = "HalfCheetah-v5"
    seed = 100
    max_timesteps = 1000000
    start_timesteps = 10000
    eval_freq = 2000
    eval_episodes =10
    expl_noise = 0.2
    batch_size = 100
    learning_rate = 1e-3
    tau = 0.005
    noise_clip = 0.5
    policy_noise = 0.2
    hidden_dim= 256

    env = gym.make(env_name) #TODO: Make this into its own module, this will become hell later
    state, _ = env.reset()
    done = False
    env.action_space.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])



    trainer = TD3_Trainer(
                    state_size=state_dim,
                    action_size=action_dim,
                    hidden_size=hidden_dim,
                    max_action=max_action,
                    learning_rate=learning_rate,
                    tau=tau,
                    noise_clip=noise_clip,
                    policy_noise=policy_noise
                )

    replay_buffer = ReplayBuffer(buffer_size=int(1e6))
    evaluations = [eval_trainer(trainer, env)]

    state, _ = env.reset()
    done = False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    for t in range(max_timesteps):
        episode_timesteps += 1

        if t < start_timesteps: #TODO: Keep this step in mind for drawing as something like this is not currently in drawio
            action = env.action_space.sample()
        else:
            noise = np.random.normal(0, max_action * expl_noise, size=action_dim)
            action = (trainer.select_action(np.array(state))+ noise)
            action = action.clip(-max_action, max_action)

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        done_bool = 0 if episode_timesteps + 1 == env.spec.max_episode_steps else float(done)
        replay_buffer.append((state, action, next_state, reward, 1 - done_bool))

        state = next_state
        episode_reward += reward

        if t >= start_timesteps:
            trainer.train(replay_buffer, batch_size)

        if done:
            print(f"Episode {episode_num+1} — Timestep {t+1} — Reward: {episode_reward:.2f}")
            state, _ = env.reset()
            done = False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        if (t + 1) % eval_freq == 0:
            evaluations.append(eval_trainer(trainer, env, eval_episodes))

if __name__ == "__main__":
    main()