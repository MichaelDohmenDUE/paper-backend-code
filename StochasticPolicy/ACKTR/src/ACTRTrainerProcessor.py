import torch

from backend.Utils.src.utils import compute_gae

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ACKTRTrainerProcessor:
    def __init__(self, actor, critic, actor_optimizer, critic_optimizer,
                 replay_buffer, vf_coef=1.0, ent_coef=0.01,
                 gamma=0.99, lam=0.95):
        self.actor = actor
        self.critic = critic
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.replay_buffer = replay_buffer
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.gamma = gamma
        self.lam = lam
        self.device = device

    def run(self):
        states, actions, _, advantages, returns = compute_gae(self.replay_buffer, gamma=self.gamma,
                                                              lam=self.lam)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Convert to tensors
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        advantages = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(returns, dtype=torch.float32, device=self.device)

        dist = self.actor(states)
        logp = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1).mean()
        values = self.critic(states).squeeze(-1)

        policy_loss = -(logp * advantages).mean()
        value_loss = (returns - values).pow(2).mean()
        loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()

        loss.backward()

        self.actor_optimizer.step()
        self.critic_optimizer.step()
