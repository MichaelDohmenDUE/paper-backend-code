from backend.PPO.continuous.src.PPO_continuous import PPOTrainer
from backend.Utils.src.BatchTransitioner import TransitionFactory
from backend.Utils.src.ReplayBuffer import ReplayBuffer
from backend.Utils.src.EnviromentHandler import EnvironmentHandler
class DataCollectionProcessor:
    def run(self, env_handler: EnvironmentHandler, trainer: PPOTrainer, replay_buffer: ReplayBuffer,
            transition_factory: TransitionFactory, rollout_size: int):
        episode_timesteps= 0
        state = env_handler.reset()
        for step in range(rollout_size):
            episode_timesteps += 1
            action, logp, value = trainer.select_action(state)
            next_state, reward, done, done_bool = env_handler.step(action, episode_timesteps)
            transition = transition_factory.create(
                state=state,
                action=action,
                logp=logp,
                reward=reward,
                done=done_bool,
                value=value
            )
            replay_buffer.append(transition)
            state = next_state

            if done:
                print(f"Rollout: {episode_timesteps}")
                state = env_handler.reset()
                episode_timesteps = 0

        _, _, last_value = trainer.select_action(state)
        return last_value