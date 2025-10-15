import numpy as np
import torch
import gym

from backend.T3D.src.TD3_Trainer import TD3_Trainer
from backend.utils import ReplayBuffer

def eval_trainer(trainer, env, eval_episodes=5):
    avg_reward = 0
    for _ in range(eval_episodes):
        state, done = env.reset(), False
        while not done:
            action = trainer.select_action(np.array(state))
            state, reward, done, _ = env.step(action)
            avg_reward += reward
    avg_reward /= eval_episodes
    print(f"Average Reward over {eval_episodes} episodes: {avg_reward:.3f}")
    return avg_reward

def main():
    env_name = "HalfCheetah-v2"
    seed = 0
    max_timesteps = int(1e4)
    start_timesteps = int(1e3)
    eval_freq = int(2e3)
    expl_noise = 0.1
    batch_size = 256


    #Seeding  TODO: Encapsulate this
    env = gym.make(env_name)
    env.seed(seed)
    env.action_space.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = 1# env.observation_space # TODO: clean this up where do I get the maximum of the acionspace.box
                    # Minima and Maxima are not where they were with gym 0.26.2
    trainer = TD3_Trainer(state_dim, action_dim, max_action, max_timesteps, batch_size, eval_freq, expl_noise)

    replay_buffer = ReplayBuffer(buffer_size=100000)
    evaluations = [eval_trainer(trainer, env)]

    state, done = env.reset(), False
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

        next_state, reward, done, terminated ,info = env.step(action)
        done_bool = float(done) if episode_timesteps < env.spec.max_episode_steps else 0

        replay_buffer.append((state, action, next_state, reward, done_bool))
        state = next_state
        episode_reward += reward

        if t >= start_timesteps:
            trainer.train(replay_buffer, batch_size)

        if done:
            print(f"Episode {episode_num+1} — Timestep {t+1} — Reward: {episode_reward:.2f}")
            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        if (t + 1) % eval_freq == 0:
            evaluations.append(eval_trainer(trainer, env))

if __name__ == "__main__":
    main()