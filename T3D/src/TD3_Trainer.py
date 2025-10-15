import torch
import torch.nn.functional as F
from torch import optim
import copy
from backend.T3D.src.Actor import Actor
from backend.T3D.src.Critic import Critic
from backend.utils import ReplayBuffer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TD3_Trainer(object):
    """
     Paper from https://arxiv.org/abs/1802.09477
    """
    def __init__(self, state_size: int, action_size: int, hidden_size: int, max_action: float, learning_rate: float, tau: float, noise_clip: float, policy_noise):
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.max_action = max_action
        self.learning_rate = learning_rate
        self.tau = tau
        self.noise_clip = noise_clip
        self.policy_noise = policy_noise
        self.syncro_frequency = 5
        self.current_episode = 0
        self.discount_factor = 0.99

        self.actor = Actor(state_size, action_size, max_action, hidden_size).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=learning_rate)

        self.critic_1 = Critic(state_size, action_size, hidden_size).to(device)
        self.critic_2 = Critic(state_size, action_size, hidden_size).to(device)
        self.critic_target_1 = copy.deepcopy(self.critic_1)
        self.critic_target_2 = copy.deepcopy(self.critic_2)

        self.optimizer_critic_1 = optim.Adam(self.critic_1.parameters(), lr=learning_rate)
        self.optimizer_critic_2 = optim.Adam(self.critic_2.parameters(), lr=learning_rate)

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()


    def train(self, replay_buffer: ReplayBuffer, batch_size: int =256):

        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        self.current_episode += 1


        with torch.no_grad():

            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)

            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)


            target_Q1 = self.critic_target_1(next_state, next_action)
            target_Q2 = self.critic_target_2(next_state, next_action)

            target_min = torch.min(target_Q1, target_Q2)
            target_min= reward + not_done * self.discount_factor * target_min


        current_Q1= self.critic_1(state, action)
        current_Q2 = self.critic_2(state, action)

        critic_loss_1 = F.mse_loss(current_Q1, target_min)
        critic_loss_2 = F.mse_loss(current_Q2, target_min)

        self.optimizer_critic_1.zero_grad()
        critic_loss_1.backward()
        self.optimizer_critic_1.step()

        self.optimizer_critic_2.zero_grad()
        critic_loss_2.backward()
        self.optimizer_critic_2.step()

        # TODO: This should probably become its own class to get it in shape with the drawn graph
        if self.current_episode % self.syncro_frequency == 0:

            current_actor_loss = -self.critic_target_1(state, self.actor(state)).mean()

            self.optimizer_actor.zero_grad()
            current_actor_loss.backward()
            self.optimizer_actor.step()

            for param, target_param in zip( self.critic_1.parameters(), self.critic_target_1.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip( self.critic_2.parameters(), self.critic_target_2.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
