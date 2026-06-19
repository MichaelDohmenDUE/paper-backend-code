from torch import nn


def preprocess(state):
    return state / 255.0


def __init__(self, action_size):
    super().__init__()
    self.conv = nn.Sequential(
        nn.Conv2d(4, 32, kernel_size=8, stride=4),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=4, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, stride=1),
        nn.ReLU(),
    )
    self.value = nn.Sequential(
        nn.Linear(64 * 7 * 7, 512),
        nn.ReLU(),
        nn.Linear(512, 1)
    )
    self.advantage= nn.Sequential(
        nn.Linear(64 * 7 * 7, 512),
        nn.ReLU(),
        nn.Linear(512, action_size)
    )


def forward(self, x):
    x = preprocess(x)
    features = self.conv(x)
    features = features.view(features.size(0), -1)
    values = self.value(features)
    advantages = self.advantage(features)
    q_values = values + (advantages - advantages.mean(dim=1, keepdim=True))

    return q_values