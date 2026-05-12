import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


def preprocess(state):
    return state / 255.0


class ACERNet(nn.Module):
    def __init__(self, action_dim):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU()
        )

        self.fc_shared = nn.Linear(64 * 7 * 7, 512)
        self.actor_head = nn.Linear(512, action_dim)
        self.critic_head = nn.Linear(512, action_dim)

    def forward(self, state):
        x = preprocess(state)
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc_shared(x))
        logits = self.actor_head(x)
        q_values = self.critic_head(x)
        return logits, q_values

    def get_policy_logits(self, state):
        logits, _ = self.forward(state)
        return logits

    def get_q_values(self, state):
        _, q_vals = self.forward(state)
        return q_vals

    def log_prob(self, state, action):
        logits, _ = self.forward(state)
        dist = Categorical(logits=logits)
        return dist.log_prob(action)