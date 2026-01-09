import torch
from torch import optim

from backend.CommonModels.src.ActorPPO import ActorPPO
from backend.CommonModels.src.CriticPPO import CriticPPO
from backend.PPO.continuous.src.ActionHandler import ActionHandler
from backend.PPO.continuous.src.DataCollectionProcessor import DataCollectionProcessor
from backend.Utils.src.BatchTransitioner import TransitionFactory
from backend.PPO.continuous.src.PPOTrainerProcessor import PPOTrainerProcessor
from backend.Utils.src.BatchTransitioner import TransitionSpec
from backend.Utils.src.EnviromentHandler import EnvironmentHandler
from backend.Utils.src.ReplayBuffer import ReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def eval_trainer(trainer, env_handler, eval_episodes=5):
    avg_reward = 0.0
    for _ in range(eval_episodes):
        state = env_handler.reset()
        done = False
        while not done:
            state_t = torch.FloatTensor(state.reshape(1, -1)).to(trainer.device)
            with torch.no_grad():
                dist = trainer.actor(state_t)
                action = dist.mean.cpu().numpy().flatten()

            next_state, reward, done, _ = env_handler.step(action, 0)
            avg_reward += reward
            state = next_state
    avg_reward /= eval_episodes
    print(f"Average Reward over {eval_episodes} episodes: {avg_reward:.3f}")
    return avg_reward

def main():
    env_name = "InvertedPendulum-v5"
    seed = 100
    rollout_size = 2048
    batch_size = 64
    epochs = 10
    num_updates = 1000
    lr = 3e-4
    hidden_dim = 64
    gamma = 0.99
    lam = 0.95

    spec = TransitionSpec(["state", "action","logp", "reward", "done","value"])
    transition_factory = TransitionFactory(spec)

    env_handler = EnvironmentHandler(env_name, seed)
    state_dim, action_dim, _ = env_handler.get_env_specs()

    actor = ActorPPO(state_dim, action_dim, hidden_dim).to(device)
    critic = CriticPPO(state_dim, hidden_dim).to(device)
    optimizer = optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=lr)

    action_handler = ActionHandler(actor, critic, device)

    replay_buffer = ReplayBuffer(spec, max_buffer_size=rollout_size, batch_size=batch_size)

    trainer = PPOTrainerProcessor(actor, critic, optimizer, replay_buffer, batch_size, epochs, gamma=gamma, lam=lam)

    data_collector = DataCollectionProcessor(env_handler, transition_factory, replay_buffer, rollout_size,
                                             action_handler)

    for update in range(num_updates):
        last_value = data_collector.run()
        trainer.run(last_value)

        if update % 10 == 0:
            eval_trainer(trainer, env_handler, eval_episodes=5)


if __name__ == "__main__":
    main()
