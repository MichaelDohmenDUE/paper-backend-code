import torch

from backend.CommonModels.src.Policy_VPG import PolicyVPG
from backend.Utils.src.EnviromentHandler import EnvironmentHandler
from backend.VGD.src.TrainerVPG import VPGTrainer


# TODO: Training is naturally unstable, keep this in mind later

def main():
    """
    Related Paper: https://link.springer.com/article/10.1007/BF00992696
    """
    learn_rate = 1e-4
    num_episodes = 2000
    seed = 42
    hidden_dim = 32
    env_name = "CartPole-v1"
    beta = 0.01
    gamma = 0.99

    env_handler = EnvironmentHandler(env_name, seed=seed)
    policy = PolicyVPG(env_handler.state_dim, env_handler.action_dim, hidden_dim=hidden_dim)
    optimizer = torch.optim.SGD(policy.parameters(), lr=learn_rate)
    trainer = VPGTrainer(policy, optimizer, beta, gamma)

    for episode in range(num_episodes):
        ep_return, ep_length = trainer.train_episode(env_handler)

        if (episode + 1) % 50 == 0:
            print(f"Ep {episode + 1}: length {ep_length}, return {ep_return}")


if __name__ == "__main__":
    main()
