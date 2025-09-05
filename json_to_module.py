import torch
from torch import nn
from collections import defaultdict

json_module = {
    "name": "Actor",
    "nodes": [
        {"name": "fc1", "type": "Linear", "input_features": 10, "output_features": 20},
        {"name": "fc2", "type": "Linear", "input_features": 20, "output_features": 4},
        {"name": "fc3", "type": "Linear", "input_features": 20, "output_features": 10},
        {"name": "fc4", "type": "Linear", "input_features": 10, "output_features": 1},
        {"name": "fc5", "type": "Linear", "input_features": 20, "output_features": 1},
        {"name": "relu", "type": "ReLU"},
        {"name": "softmax", "type": "Softmax"},
        {"name": "tanh", "type": "Tanh"}
    ],
    "inputs": "fc1",
    "connections": [
        {"from": "fc1", "to": "relu"},
        {"from": "relu", "to": "fc2"},
        {"from": "relu", "to": "fc3"},
        {"from": "fc2", "to": "softmax"},
        {"from": "fc1", "to": "tanh"},
        {"from": "fc3", "to": "fc4"},
        {"from": "tanh", "to": "fc5"},
        {"from": "fc3", "to": "softmax"},
    ]

}

layer_maps = {
    "Linear": nn.Linear,
    "ReLU": nn.ReLU,
    "Softmax": nn.Softmax,
    "Tanh": nn.Tanh
}


def dfs(node, graph, path, streams):
    path.append(node)
    if node not in graph:
        streams.append(path.copy())
    else:
        for node in graph[node]:
            dfs(node, graph, path, streams)
    path.pop()


class JSONModule(nn.Module):
    def __init__(self, data):
        super().__init__()

        # initialize layers
        layers = {}
        for node in data["nodes"]:
            name = node["name"]
            layer = layer_maps[node["type"]]
            if "input_features" in node.keys():
                input_features = node["input_features"]
                output_features = node["output_features"]
                layer = layer(input_features, output_features)
            else:
                layer = layer()

            layers[name] = layer

        for key, value in layers.items():
            setattr(self, key, value)

        connections = json_module["connections"]

        # Prepare forward pass
        graph = defaultdict(list)
        for conn in connections:
            graph[conn["from"]].append(conn["to"])

        # maybe i dont even need seperate sequences, since that would duplicate the nodes
        input_layer = json_module["inputs"]
        path = []
        streams = []
        dfs(input_layer, graph, path, streams)


        forward_streams = []
        for stream in streams:
            forward_stream = []
            for node in stream:
                forward_stream.append(self.__getattr__(node))
            forward_streams.append(forward_stream)
        print(forward_streams)



if __name__ == '__main__':
    net = JSONModule(json_module)
    print(net)
