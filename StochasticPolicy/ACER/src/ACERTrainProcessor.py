from backend.Utils.src.utils import synchronize


class ACERTrainProcessor:
    def __init__(self, trainer, buffer, seq_len, replay_ratio, batch_size, tau):
        self.trainer = trainer
        self.buffer = buffer
        self.seq_len = seq_len
        self.replay_ratio = replay_ratio
        self.batch_size = batch_size
        self.tau = tau

    def run(self):
        warmup = 1000
        if len(self.buffer) < max(self.seq_len, warmup):
            return

        self.trainer.train(self.buffer, batch_size=1, on_policy=True)
        #synchronize(self.trainer.actor, self.trainer.trust_region_actor, tau=self.tau)
        for _ in range(self.replay_ratio):
            self.trainer.train(self.buffer, batch_size=self.batch_size, on_policy=False)
            #synchronize(self.trainer.actor, self.trainer.trust_region_actor, tau=self.tau)
