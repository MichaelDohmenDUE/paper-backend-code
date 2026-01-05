from backend.DDQN.ddqn_graph import build_ddqn_graph
from backend.Utils.src.NodeLib.Node import Graph


class TrainProcessor:
    def __init__(self, buffer, behavior_net, target_net, optimizer, gamma, device):
        self.buffer = buffer
        self.behavior_net = behavior_net.to(device)
        self.target_net = target_net.to(device)
        self.optimizer = optimizer
        self.gamma = gamma
        self.device = device

        # Build the DDQN graph
        self.graph = Graph(build_ddqn_graph())

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

        self.graph.run(context)
