from dataclasses import dataclass
import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim
import onnx
from onnx import numpy_helper

@dataclass
class NeuralNetwork:
    graph: nx.DiGraph
    num_layers: int

    def get_input_nodes(self):
        return self.get_nodes_of_layer(0)

    def get_output_node(self):
        """
        We are working with a single output node for now.
        """
        return self.get_nodes_of_layer(self.num_layers)[0]

    def get_nodes_of_layer(self, layer: int):
        return [n for n, attr in self.graph.nodes(data=True) if attr["layer"] == layer]

    def get_node(self, node_id):
        return self.graph.nodes[node_id]

    def in_edges(self, node_id):
        return self.graph.in_edges(node_id, data=True)


def torch_model_to_graph(torch_model):
    state_dict = torch_model.state_dict()
    G = nx.DiGraph()

    # Keep track of node ids so we can insert edges more easily later.
    node_ids = [[]]

    # Insert the input nodes (bias=0).
    input_weights = list(state_dict.items())[0][1].tolist()
    num_input_nodes = len(input_weights[0])

    node_id = 0
    for i in range(0, num_input_nodes):
        node_id += 1
        node_name = f"input.{i}"

        G.add_node(node_name, layer=0, bias=0)
        node_ids[0].append(node_name)

    # Then go on with inserting each hidden and output node.
    layer = 0
    total_number_of_layers = len(state_dict.items()) // 2
    for name, values in state_dict.items():
        # state_dict alternates between weight and bias tensors.
        if "bias" not in name:
            continue

        layer += 1
        node_id = 0
        layer_name = f"layer_{layer}" if layer < total_number_of_layers else "output"
        node_ids.append([])

        for i, bias in enumerate(values.tolist()):
            node_name = f"{layer_name}.{node_id}"

            G.add_node(f"{layer_name}.{node_id}", layer=layer, bias=bias)
            node_ids[layer].append(node_name)

            node_id += 1

    # And now the edges and their weights. Assumes fully connected network.
    layer = 0
    for name, values in state_dict.items():
        if "weight" not in name:
            continue

        # Each weight tensor has a list for each node in the next layer. The
        # elements of this list correspond to the nodes of the current layer.
        weight_tensor = values.tolist()
        for from_index, from_node in enumerate(node_ids[layer]):
            for to_index, to_node in enumerate(node_ids[layer + 1]):
                weight = weight_tensor[to_index][from_index]
                G.add_edge(from_node, to_node, weight=weight)

        layer += 1

    return NeuralNetwork(G, total_number_of_layers)


def onnx_model_to_graph(onnx_model_path: str):
    model = onnx.load(onnx_model_path)
    graph = model.graph

    initializers = {}
    for init in graph.initializer:
        initializers[init.name] = numpy_helper.to_array(init)

    gemm_nodes = [n for n in graph.node if n.op_type == "Gemm"]
    total_number_of_layers = len(gemm_nodes)

    G = nx.DiGraph()
    node_ids = [[]]

    first_weight = initializers[gemm_nodes[0].input[1]]
    num_input_nodes = first_weight.shape[1]

    for i in range(num_input_nodes):
        node_name = f"input.{i}"
        G.add_node(node_name, layer=0, bias=0)
        node_ids[0].append(node_name)

    for layer_idx, gemm_node in enumerate(gemm_nodes):
        biases = initializers[gemm_node.input[2]].tolist()
        layer_name = (
            f"layer_{layer_idx + 1}"
            if layer_idx < total_number_of_layers - 1
            else "output"
        )
        node_ids.append([])

        for i, bias in enumerate(biases):
            node_name = f"{layer_name}.{i}"
            G.add_node(node_name, layer=layer_idx + 1, bias=bias)
            node_ids[layer_idx + 1].append(node_name)

    for layer_idx, gemm_node in enumerate(gemm_nodes):
        weight_tensor = initializers[gemm_node.input[1]].tolist()
        for from_index, from_node in enumerate(node_ids[layer_idx]):
            for to_index, to_node in enumerate(node_ids[layer_idx + 1]):
                weight = weight_tensor[to_index][from_index]
                G.add_edge(from_node, to_node, weight=weight)

    return NeuralNetwork(G, total_number_of_layers)


def train_sample_network(
    num_inputs, num_hidden_nodes_per_layer, num_hidden_layers, seed=123
):
    torch.manual_seed(seed)

    # Input layer
    layers = [nn.Linear(num_inputs, num_hidden_nodes_per_layer)]

    # First n-1 hidden layers
    for _ in range(0, num_hidden_layers - 1):
        layers.append(nn.ReLU())
        layers.append(nn.Linear(num_hidden_nodes_per_layer, num_hidden_nodes_per_layer))

    # Last hidden to output layer.
    layers.append(nn.ReLU())
    layers.append(nn.Linear(num_hidden_nodes_per_layer, 1))

    model = nn.Sequential(*layers)

    # Learn z = x + y + ...
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(10):
        inputs = torch.randn(100, num_inputs)
        targets = inputs.sum(dim=1, keepdim=True)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()

    return model
