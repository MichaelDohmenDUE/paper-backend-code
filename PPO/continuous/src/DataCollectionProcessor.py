from backend.PPO.continuous.src.ActionHandler import ActionHandler
from backend.Utils.src.BatchTransitioner import TransitionFactory
from backend.Utils.src.ReplayBuffer import ReplayBuffer
from backend.Utils.src.EnviromentHandler import EnvironmentHandler

class DataCollectionProcessor:
    def __init__(self, env_handler: EnvironmentHandler, transition_factory: TransitionFactory,
                 replay_buffer: ReplayBuffer, rollout_size: int,
                 action_handler: ActionHandler):
        self.env_handler = env_handler
        self.transition_factory = transition_factory
        self.replay_buffer = replay_buffer
        self.rollout_size = rollout_size
        self.policy = action_handler

    def run(self):
        self.replay_buffer.buffer.clear()
        episode_timesteps= 0
        state = self.env_handler.reset()
        for step in range(self.rollout_size):
            episode_timesteps += 1
            action, logp, value = self.policy.select_action(state)
            next_state, reward, done, done_bool = self.env_handler.step(action, episode_timesteps)
            transition = self.transition_factory.create(
                state=state,
                action=action,
                logp=logp,
                reward=reward,
                done=done_bool,
                value=value,
                bootstrap_value=None
            )
            self.replay_buffer.append(transition)
            state = next_state

            if done:
                #print(f"Rollout: {episode_timesteps}")
                state = self.env_handler.reset()
                episode_timesteps = 0

        _, _, last_value = self.policy.select_action(state)
        self.replay_buffer.buffer[-1].bootstrap_value = last_value # overwrite last transition Bootsrap value that gets later extracted for GAE