from backend.Utils.src.GlobalCounter import GlobalCounter
from backend.Utils.src.NodeLib.NodeLibrary import *
from backend.Utils.src.ReplayBuffer import ReplayBuffer

class TrainProcessor:
    """
    Twin Delayed Deep Deterministic Policy Gradient (TD3)
    Paper: https://arxiv.org/abs/1802.09477
    """


    def __init__(self, actor: nn.Module, critic_1: nn.Module, critic_2: nn.Module,
                 optimizer_critic_1: torch.optim.Optimizer, optimizer_critic_2: torch.optim.Optimizer,
                 optimizer_actor: torch.optim.Optimizer,
                 actor_target: nn.Module, critic_target_1: nn.Module, critic_target_2: nn.Module,
                 replay_buffer: ReplayBuffer, global_counter: GlobalCounter, max_action: float, learning_rate: float,
                 noise_clip: float,
                 policy_noise, start_timesteps=25000, synchro_frequency: int = 2, discount_factor: float = 0.99,
                 device: torch.device = torch.device("cpu")):
        self.max_action = max_action
        self.learning_rate = learning_rate
        self.noise_clip = noise_clip
        self.policy_noise = policy_noise
        self.syncro_frequency = synchro_frequency
        self.discount_factor = discount_factor
        self.global_counter = global_counter
        self.start_timesteps = start_timesteps
        self.actor = actor
        self.critic_1 = critic_1
        self.critic_2 = critic_2
        self.optimizer_critic_1 = optimizer_critic_1
        self.optimizer_critic_2 = optimizer_critic_2
        self.optimizer_actor = optimizer_actor
        self.actor_target = actor_target
        self.critic_target_1 = critic_target_1
        self.critic_target_2 = critic_target_2
        self.replay_buffer = replay_buffer
        self.device = device

    def old_update_actor(self, state: torch.Tensor):
        actor_loss = None
        if self.global_counter.get() % self.syncro_frequency == 0:
            action = self.actor(state)
            q_val = self.critic_1(state, action).squeeze()
            actor_loss = deterministic_policy_gradient(q_val)
            optimizer_update(optimizer=self.optimizer_actor, loss=actor_loss)
            return actor_loss.item()
        return None

    def run(self):
        if self.global_counter.get() >= self.start_timesteps:
            return self.train()
        return {}

    def train(self):
        batch = self.replay_buffer.sample_batch()
        state, action, reward, next_state, done = detransition(self.replay_buffer.spec.fields, batch, self.device)
        with torch.no_grad():
            next_action = self.actor_target(next_state)
            next_action = action_with_gaussian_noise(next_action, self.policy_noise, self.noise_clip, self.max_action)
            next_action = clipper(next_action, self.max_action)

            target_Q1 = self.critic_target_1(next_state, next_action).squeeze()
            target_Q2 = self.critic_target_2(next_state, next_action).squeeze()

            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = bellman(target_Q, reward, done, self.discount_factor)

        current_Q1 = self.critic_1(state, action).squeeze()
        current_Q2 = self.critic_2(state, action).squeeze()

        critic_loss_1 = mean_squared_error(current_Q1, target_Q)
        critic_loss_2 = mean_squared_error(current_Q2, target_Q)

        optimizer_update(optimizer=self.optimizer_critic_1, loss=critic_loss_1)
        optimizer_update(optimizer=self.optimizer_critic_2, loss=critic_loss_2)
        #Actor_Update
        action = self.actor(state)
        q_val = self.critic_1(state, action).squeeze()
        actor_loss = deterministic_policy_gradient(q_val)
        actor_loss_val = timed_optimizer_update(self.optimizer_actor, loss=actor_loss,
                                                gloabal_step=self.global_counter.get(),
                                                syncro_frequency=self.syncro_frequency)

        metrics = {
            "losses/critic_loss 1": critic_loss_1.item(),
            "losses/critic_loss 2": critic_loss_2.item(),
            "losses/actor_loss": actor_loss_val,
        }

        return metrics
