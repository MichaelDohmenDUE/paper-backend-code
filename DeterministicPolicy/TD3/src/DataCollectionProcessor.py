from backend.DeterministicPolicy.TD3.src.ActionHandler import ActionHandler
from backend.Utils.src.BatchTransitioner import TransitionFactory
from backend.Utils.src.EnviromentHandler import EnvironmentHandler
from backend.Utils.src.GlobalCounter import GlobalCounter
from backend.Utils.src.NodeLib.NodeLibrary import reset_handler
from backend.Utils.src.ReplayBuffer import ReplayBuffer


class DataCollectionProcessor:
    def __init__(self, env_handler: EnvironmentHandler, action_handler: ActionHandler,
                 transition_factory: TransitionFactory, replay_buffer: ReplayBuffer,
                 global_counter: GlobalCounter):
        self.env = env_handler
        self.action_handler = action_handler
        self.factory = transition_factory
        self.buffer = replay_buffer

        self.state = self.env.reset()
        self.episode_reward = 0
        self.episode_timesteps = 0
        self.episode_num = 0
        self.global_counter = global_counter

    def run(self):
        action = self.action_handler.select_action(self.state, self.global_counter.get())
        next_state, reward, truncated, terminated, info = self.env.step_ddpg(action)

        transition = self.factory.forward(state=self.state, action=action, reward=reward, next_state=next_state,
                                          done=terminated)

        self.buffer.append(transition)
        self.state = reset_handler(env=self.env, next_state=next_state, done=terminated or truncated)
        #Logging
        self.episode_reward += reward
        self.global_counter.set(self.global_counter.get() + 1)
        self.episode_timesteps += 1
        metrics = {}
        if truncated or terminated:
            print(f"Episode {self.episode_num + 1} — Reward: {self.episode_reward:.2f}")
            metrics = {
                "charts/episodic_return": self.episode_reward,
                "charts/episodic_length": self.episode_timesteps,
                "global_step": self.global_counter,
            }
            self.episode_reward = 0
            self.episode_timesteps = 0
            self.episode_num += 1
        return metrics
