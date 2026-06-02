import geometrical.network as network
import geometrical.pycad as pycad
import numpy as np
import concurrent.futures


def _process_hidden_node(nn, hidden_node, pwls_by_node_id):
    bias = nn.get_node(hidden_node)["bias"]

    incoming_pwls = []
    for in_id, _, data in nn.in_edges(hidden_node):
        weight = data["weight"]
        pwl = pycad.scale(pwls_by_node_id[in_id], weight)
        incoming_pwls.append(pwl)

    pwl = pycad.sum_pwls(incoming_pwls)
    pycad.add_bias_in_place(pwl, bias)

    return hidden_node, pycad.apply_relu(pwl)


def construct_network_pwl(nn: network.NeuralNetwork, parallel=True):
    # Input nodes
    pwls_by_node_id = {}
    input_nodes = nn.get_input_nodes()
    component_function_dims = len(input_nodes) + 2  # 2 because of a0 and z

    for dim, node in enumerate(input_nodes, start=1):
        component_function = np.zeros(component_function_dims, dtype=np.float32)
        component_function[-1] = -1
        component_function[dim] = 1

        input_pwl = pycad.create_pwl_from_constraints(
            [], component_function, len(input_nodes)
        )
        pwls_by_node_id[node] = input_pwl

    if parallel:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            # Hidden layers.
            for layer in range(1, nn.num_layers):
                layer_nodes = nn.get_nodes_of_layer(layer)

                def process(hidden_node):
                    return _process_hidden_node(nn, hidden_node, pwls_by_node_id)

                for node, final_pwl in executor.map(process, layer_nodes):
                    pwls_by_node_id[node] = final_pwl

            # Output node.
            output_node = nn.get_output_node()
            bias = nn.get_node(output_node)["bias"]

            incoming_pwls = []
            for incoming_node, _, data in list(nn.in_edges(output_node)):
                pwl = pwls_by_node_id[incoming_node]
                pycad.scale_in_place(pwl, data["weight"])
                incoming_pwls.append(pwl)

            def sum_pair(pair):
                a, b = pair
                return pycad.sum_pwls([a, b])

            # Reduce pairwise in parallel. E.g. for PWL IDs [1,2,3,4] does a sum
            # of [1,2] and [3,4] in parallel and then sums the results.
            work = incoming_pwls
            while len(work) > 1:
                pairs = []
                carry = None
                it = iter(work)
                for a in it:
                    b = next(it, None)
                    if b is None:
                        carry = a
                        break
                    pairs.append((a, b))

                merged = list(executor.map(sum_pair, pairs)) if pairs else []
                if carry is not None:
                    merged.append(carry)
                work = merged

            if not work:
                raise ValueError("Output node has no incoming edges; cannot construct PWL.")
            pwl = work[0]
            pycad.add_bias_in_place(pwl, bias)

            return pwl
    else:
        for layer in range(1, nn.num_layers):
            for hidden_node in nn.get_nodes_of_layer(layer):
                _, pwl = _process_hidden_node(nn, hidden_node, pwls_by_node_id)
                pwls_by_node_id[hidden_node] = pwl
        # Output node (serial)
        output_node = nn.get_output_node()
        bias = nn.get_node(output_node)["bias"]

        incoming_pwls = []
        for incoming_node, _, data in list(nn.in_edges(output_node)):
            pwl = pwls_by_node_id[incoming_node]
            pwl = pycad.scale(pwl, data["weight"])
            incoming_pwls.append(pwl)

        pwl = pycad.sum_pwls(incoming_pwls)
        pycad.add_bias_in_place(pwl, bias)

        return pwl
