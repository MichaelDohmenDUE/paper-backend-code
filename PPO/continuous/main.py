import torch
from torch import optim

from backend.CommonModels.src.ActorPPO import ActorPPO
from backend.CommonModels.src.CriticPPO import CriticPPO
from backend.PPO.continuous.src.ActionHandler import ActionHandler
from backend.PPO.continuous.src.DataCollectionProcessor import DataCollectionProcessor
from backend.Utils.src.BatchTransitioner import TransitionFactory
from backend.PPO.continuous.src.PPO_continuous import PPOTrainer
from backend.Utils.src.BatchTransitioner import TransitionSpec
from backend.Utils.src.EnviromentHandler import EnvironmentHandler
from backend.Utils.src.EvaluationHelper import eval_trainer
from backend.Utils.src.ReplayBuffer import ReplayBuffer
from backend.Utils.src.utils import compute_gae

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_update(trainer, replay_buffer, last_value, batch_size, epochs):
    states, actions, logps, advs, rets = compute_gae(replay_buffer, gamma=0.99, lam=0.95, last_value=last_value)
    stats = trainer.train(states, actions, logps, advs, rets, batch_size=batch_size, epochs=epochs)
    replay_buffer.buffer.clear()
    return stats


def main():
    env_name = "InvertedPendulum-v5"
    seed = 100
    rollout_size = 2048
    batch_size = 64
    epochs = 10
    num_updates = 1000
    lr = 3e-4
    hidden_dim = 64

    spec = TransitionSpec(["state", "action","logp", "reward", "done","value"])
    transition_factory = TransitionFactory(spec)

    env_handler = EnvironmentHandler(env_name, seed)
    state_dim, action_dim, _ = env_handler.get_env_specs()

    actor = ActorPPO(state_dim, action_dim, hidden_dim).to(device)
    critic = CriticPPO(state_dim, hidden_dim).to(device)
    optimizer = optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=lr)

    action_handler = ActionHandler(actor, critic, device)

    trainer = PPOTrainer(actor, critic, optimizer)

    replay_buffer = ReplayBuffer(spec, max_buffer_size=rollout_size, batch_size=batch_size)

    data_collector = DataCollectionProcessor(env_handler, transition_factory, replay_buffer, rollout_size,
                                             action_handler)

    for update in range(num_updates):
        last_value = data_collector.run()

        train_update(trainer, replay_buffer, last_value, batch_size, epochs)

        #if update % 10 == 0:
        #    eval_trainer(trainer, env_handler, eval_episodes=5)


if __name__ == "__main__":
    main()
