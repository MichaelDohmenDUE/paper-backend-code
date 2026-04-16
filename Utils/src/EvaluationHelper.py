import torch


def eval_trainer(trainer, env_handler, eval_episodes=5):
    avg_reward = 0.0
    for _ in range(eval_episodes):
        state = env_handler.reset()
        done = False
        while not done:
            state_t = torch.FloatTensor(state.reshape(1, -1)).to(trainer.device)
            with torch.no_grad():
                action = trainer.actor(state_t).cpu().numpy().flatten()

            next_state, reward, done, _ = env_handler.step(action)
            avg_reward += reward
            state = next_state
    avg_reward /= eval_episodes
    print(f"Average Reward over {eval_episodes} episodes: {avg_reward:.3f}")
    return avg_reward
