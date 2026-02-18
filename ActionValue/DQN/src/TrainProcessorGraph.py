class TrainProcessor:
    def __init__(self, graph, buffer, behavior_net, target_net, optimizer, gamma, device):
        self.buffer = buffer
        self.behavior_net = behavior_net.to(device)
        self.target_net = target_net.to(device)
        self.optimizer = optimizer
        self.gamma = gamma
        self.device = device
        self.graph = graph

    def run(self):
        if len(self.buffer) < self.buffer.batch_size:
            return

        context = {
            "buffer": self.buffer,
            "behavior_net": self.behavior_net,
            "target_net": self.target_net,
            "optimizer": self.optimizer,
            "gamma": self.gamma,
            "device": self.device,
        }

        if self.graph.validate(context):
            self.graph.run(context)
