from backend.AbstractHandlers.AbstractActionHandler import AbstractActionHandler
from backend.Utils.src import ReplayBuffer
from backend.Utils.src.BatchTransitioner import TransitionFactory
from backend.Utils.src.EnviromentHandler import EnvironmentHandler


class DataCollectionProcessor:
    def __init__(self, env_handler: EnvironmentHandler, transition_factory: TransitionFactory,
                 replay_buffer: ReplayBuffer, action_handler: AbstractActionHandler):
        self.env_handler = env_handler
        self.action_handler = action_handler
        self.transition_factory = transition_factory
        self.replay_buffer = replay_buffer
        self.policy = action_handler

    def run(self):
        self.replay_buffer.buffer.clear()
        episode_timesteps = 0
        state = self.env_handler.reset()
        done = False

        while not done:
            action, log_prob = self.action_handler.select_action(state)
            next_state, reward, done, done_bool = self.env_handler.step(action, episode_timesteps)
            transition = self.transition_factory.create(
                state=state,
                action=action,
                logp=log_prob,
                reward=reward,
                done=done_bool
            )
            self.replay_buffer.append(transition)
            state = next_state
            episode_timesteps += 1