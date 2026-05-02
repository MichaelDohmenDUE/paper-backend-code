from collections import deque
from enum import Enum

import torch

class Signal(Enum):
    NOSIGNAL = "NOSIGNAL"

class Node:
    def __init__(self, name: str, inputs: list[str], outputs: list[str], function=None, no_grad: bool = False):
        self.name = name
        self.function = function
        self.inputs = inputs
        self.outputs = outputs
        self.no_grad = no_grad

    def forward(self, *args):
        if self.function is not None:
            return self.function(*args)
        raise NotImplementedError("Implement the function, you silly goose")

    def __call__(self, context: dict):
        args = [context[key] for key in self.inputs]
        if any(arg is Signal.NOSIGNAL for arg in args):
            result = Signal.NOSIGNAL
        else:
            if self.no_grad:
                with torch.no_grad():
                    result = self.forward(*args)
            else:
                result = self.forward(*args)

        if result is Signal.NOSIGNAL:
            for key in self.outputs:
                context[key] = Signal.NOSIGNAL
            return None
        if len(self.outputs) == 0:
            return None
        if len(self.outputs) == 1:
            context[self.outputs[0]] = result
        else:
            for key, value in zip(self.outputs, result):
                context[key] = value
        return None

class Graph:
    def __init__(self, nodes, initial_keys: list[str]):
        self.nodes: list[Node] = nodes
        self.execution_order_list: list[Node] = []

        self._compile(initial_keys)

    def run(self, context: dict):
        for node in self.execution_order_list:
            node(context)
        return context

    def _compile(self, initial_keys: list[str]):
        output_to_node = {}
        for node in self.nodes:
            for output in node.outputs:
                if output in output_to_node:
                    raise ValueError(f"Error:Duplicate '{output}' from {node.name} , {output_to_node[output].name}")
                output_to_node[output] = node

        grad = {node.name: 0 for node in self.nodes}
        adjacent_nodes = {node.name: [] for node in self.nodes}
        provided_keys = set(initial_keys)

        for node in self.nodes:
            for node_input in node.inputs:
                if node_input in provided_keys:
                    continue
                elif node_input in output_to_node:
                    parent = output_to_node[node_input]
                    adjacent_nodes[parent.name].append(node)
                    grad[node.name] += 1
                else:
                    raise ValueError(
                        f"Error: node_input '{node_input}' for node '{node.name}' not here.")

        queue = deque([n for n in self.nodes if grad[n.name] == 0])
        sorted_nodes = []

        while queue:
            current = queue.popleft()
            sorted_nodes.append(current)
            for child in adjacent_nodes[current.name]:
                grad[child.name] -= 1
                if grad[child.name] == 0:
                    queue.append(child)

        if len(sorted_nodes) != len(self.nodes):
            unresolved = [n.name for n in self.nodes if grad[n.name] > 0]
            raise ValueError(f"Error - Circle found : {unresolved}")

        self.execution_order_list = sorted_nodes


class PropsNode(Node):
    def __init__(self, name, inputs, outputs, props=None, function=None, no_grad=False):
        combined = (props if props else []) + inputs
        super().__init__(name, combined, outputs, function, no_grad)

    def forward(self, *args):
        if self.function is not None:
            return self.function(*args)
        raise NotImplementedError("Implement the function, you silly goose")

    def __call__(self, context: dict):
        args = [context[key] for key in self.inputs]
        if self.no_grad:
            with torch.no_grad():
                result = self.forward(*args)
        else:
            result = self.forward(*args)
        # Special Signal to abort Graph Execution, different from "None" as an output
        if result is Signal.NOSIGNAL:
            return Signal.NOSIGNAL

        if len(self.outputs) == 0:
            return None
        if len(self.outputs) == 1:
            context[self.outputs[0]] = result
        else:
            for key, value in zip(self.outputs, result):
                context[key] = value
        return None

class Graph:
    def __init__(self, nodes, initial_keys: list[str]):
        self.nodes: list[Node] = nodes
        self.execution_order_list: list[Node] = []

        self._compile(initial_keys)

    def run(self, context: dict):
        for node in self.execution_order_list:
            signal = node(context)
            if signal is Signal.NOSIGNAL:
                return None
        return context

    def _compile(self, initial_keys: list[str]):
        output_to_node = {}
        for node in self.nodes:
            for output in node.outputs:
                if output in output_to_node:
                    raise ValueError(f"Error:Duplicate '{output}' from {node.name} , {output_to_node[output].name}")
                output_to_node[output] = node

        grad = {node.name: 0 for node in self.nodes}
        adjacent_nodes = {node.name: [] for node in self.nodes}
        provided_keys = set(initial_keys)

        for node in self.nodes:
            for node_input in node.inputs:
                if node_input in provided_keys:
                    continue
                elif node_input in output_to_node:
                    parent = output_to_node[node_input]
                    adjacent_nodes[parent.name].append(node)
                    grad[node.name] += 1
                else:
                    raise ValueError(
                        f"Error: node_input '{node_input}' for node '{node.name}' not here.")

        queue = deque([n for n in self.nodes if grad[n.name] == 0])
        sorted_nodes = []

        while queue:
            current = queue.popleft()
            sorted_nodes.append(current)
            for child in adjacent_nodes[current.name]:
                grad[child.name] -= 1
                if grad[child.name] == 0:
                    queue.append(child)

        if len(sorted_nodes) != len(self.nodes):
            unresolved = [n.name for n in self.nodes if grad[n.name] > 0]
            raise ValueError(f"Error - Circle found : {unresolved}")

        self.execution_order_list = sorted_nodes