import torch

from backend.CommonModels.src.Policy import Policy
from backend.DDQN.src.DDQNTrainer import DDQNTrainer
from backend.Utils.src.EnviromentHandler import EnvironmentHandler
from backend.Utils.src.ReplayBuffer import ReplayBuffer

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    environment = "CartPole-v1"
    seed = 0
    gamma = 0.99
    batch_size = 64
    update_frq = 40
    epsilon = 0.2
    hidden_size = 32
    num_episodes = 600

    env_handler = EnvironmentHandler(environment, seed)


    behavior_policy = Policy(env_handler.state_dim, env_handler.action_dim, hidden_size).to(device)
    target_policy = Policy(env_handler.state_dim, env_handler.action_dim, hidden_size).to(device)
    target_policy.load_state_dict(behavior_policy.state_dict())

    optimizer = torch.optim.Adam(behavior_policy.parameters())
    buffer = ReplayBuffer()


    trainer = DDQNTrainer(
        env_handler=env_handler,
        behavior_policy=behavior_policy,
        target_policy=target_policy,
        optimizer=optimizer,
        buffer=buffer,
        gamma=gamma,
        batch_size=batch_size,
        update_freq=update_frq,
        epsilon=epsilon,
        device=device
    )

    trainer.train(num_episodes=num_episodes)