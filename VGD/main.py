from backend.Utils.src.BatchTransitioner import TransitionFactory, TransitionSpec
from backend.Utils.src.ReplayBuffer import ReplayBuffer
from backend.VGD.src.ActionHandler import ActionHandler
import torch

from backend.CommonModels.src.Policy_VPG import PolicyVPG
from backend.Utils.src.EnviromentHandler import EnvironmentHandler
from backend.VGD.src.DataCollector import DataCollectionProcessor
from backend.VGD.src.TrainerVPG import VPGTrainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

    spec = TransitionSpec(["state", "action", "logp", "reward", "done"])
    transition_factory = TransitionFactory(spec)
    replay_buffer = ReplayBuffer(spec, max_buffer_size=500, batch_size=1)

    env_handler = EnvironmentHandler(env_name, seed=seed)

    observation_size, action_size, _ = env_handler.get_env_specs()
    policy = PolicyVPG(observation_size, action_size, hidden_dim=hidden_dim).to(device)

    action_handler = ActionHandler(policy, device)
    optimizer = torch.optim.SGD(policy.parameters(), lr=learn_rate)

    data_collector = DataCollectionProcessor(env_handler, transition_factory, replay_buffer, action_handler)

    trainer = VPGTrainer(policy, replay_buffer, action_handler, optimizer, beta, gamma, device=device)

    for episode in range(num_episodes):
        data_collector.run()
        ep_return, ep_length = trainer.run()

        if (episode + 1) % 50 == 0:
            print(f"Ep {episode + 1}: length {ep_length}, return {ep_return}")


if __name__ == "__main__":
    main()
