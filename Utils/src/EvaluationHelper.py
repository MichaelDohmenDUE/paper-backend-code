def eval_trainer(trainer, env_handler, eval_episodes=5):
    avg_reward = 0.0
    for _ in range(eval_episodes):
        state = env_handler.reset()
        done = False
        while not done:
            result = trainer.select_action(state)
            action = result[0] if isinstance(result, (tuple, list)) else result

            next_state, reward, done, _ = env_handler.step(action, 0)
            avg_reward += reward
            state = next_state
    avg_reward /= eval_episodes
    print(f"Average Reward over {eval_episodes} episodes: {avg_reward:.3f}")
    return avg_reward
