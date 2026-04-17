from backend.Utils.src.utils import synchronize


class ACERTrainProcessor:
    def __init__(self, trainer, buffer, seq_len, replay_ratio, batch_size, tau):
        self.trainer = trainer
        self.buffer = buffer
        self.seq_len = seq_len
        self.replay_ratio = replay_ratio
        self.batch_size = batch_size
        self.tau = tau

    def run(self, on_policy_rollouts):
        if len(on_policy_rollouts) == 0:
            return

        self.trainer.train(on_policy_rollouts, batch_size=len(on_policy_rollouts), on_policy=True)

        for _ in range(self.replay_ratio):
            self.trainer.train(self.buffer, batch_size=self.batch_size, on_policy=False)
