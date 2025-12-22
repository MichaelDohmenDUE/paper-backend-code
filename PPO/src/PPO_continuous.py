import numpy as np
import torch
from torch import nn, optim

from backend.CommonModels.src.ActorPPO import ActorPPO
from backend.CommonModels.src.CriticPPO import CriticPPO

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PPOTrainer:
    # TODO : Make PPO Trainer agnostic between continuous and discrete
    def __init__(self, state_dim, action_dim, hidden_dim=64, lr=3e-4, clip_eps=0.2, vf_coef=1.0, ent_coef=0.01, max_grad_norm=0.5):
        self.actor = ActorPPO(state_dim, action_dim, hidden_dim).to(device)
        self.critic = CriticPPO(state_dim, hidden_dim).to(device)
        self.optimizer = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr)
        self.clip_eps = clip_eps
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.use_value_clip = False # TODO: Spinup Implementation uses value Clipping, original PPO Paper does not
        self.device = device

    def select_action(self, state):
        state = np.array(state, dtype=np.float32)
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            dist = self.actor(state_t)
            action = dist.sample()
            #print(action)
            logp = dist.log_prob(action).sum(-1)
            value = self.critic(state_t).squeeze(-1)
        return action.cpu().numpy().squeeze(0), float(logp.cpu().numpy()), float(value.cpu().numpy())

    def train(self, states, actions, old_logps, advantages, returns,batch_size=64, epochs=10):
        #advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Convert to tensors
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        old_logps = torch.as_tensor(old_logps, dtype=torch.float32, device=self.device)
        advantages = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(returns, dtype=torch.float32, device=self.device)

        # Replaybuffer Rollout
        replaybuffer_rollout = list(zip(states, actions, old_logps, advantages, returns))

        for _ in range(epochs):
            np.random.shuffle(replaybuffer_rollout)
            for start in range(0, len(replaybuffer_rollout), batch_size):
                batch = replaybuffer_rollout[start:start + batch_size]
                b_states, b_actions, b_old_logps, b_adv, b_ret = zip(*batch)
                b_states = torch.stack(b_states)
                b_actions = torch.stack(b_actions)
                b_old_logps = torch.stack(b_old_logps)
                b_adv = torch.stack(b_adv)
                b_ret = torch.stack(b_ret)

                dist = self.actor(b_states)
                new_logp = dist.log_prob(b_actions).sum(-1)
                print(new_logp)
                entropy = dist.entropy().sum(-1).mean()
                value_pred = self.critic(b_states).squeeze(-1)

                # Policy Losses
                ratio = torch.exp(new_logp - b_old_logps)
                surrogate_objective = ratio * b_adv
                surrogate_objective2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * b_adv
                policy_loss = -torch.min(surrogate_objective, surrogate_objective2).mean()

                # value loss TODO: SpinUp Implementation uses value clipping original PPO does not

                value_loss = (b_ret - value_pred).pow(2).mean()

                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy

                # Backprop
                self.optimizer.zero_grad()
                loss.backward()
                #nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()),self.max_grad_norm)
                self.optimizer.step()
