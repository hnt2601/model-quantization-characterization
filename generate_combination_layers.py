import onnx
import argparse
import os
import pickle

from pathlib import Path
from tqdm import tqdm
from itertools import combinations


def get_args():
    parser = argparse.ArgumentParser(description="Test")

    parser.add_argument(
        "--onnx_path",
        type=str,
        default="./pretrained/model.onnx",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
    )
    parser.add_argument("--only_weight", action="store_true", default=False, help="")

    args = parser.parse_args()

    return args


def split_increasing_sublists(lst):
    if not lst:
        return []

    sublists = []
    current_sublist = [lst[0]]

    for i in range(1, len(lst)):
        if lst[i] == lst[i - 1] + 1:
            current_sublist.append(lst[i])
        else:
            sublists.append(current_sublist)
            current_sublist = [lst[i]]

    sublists.append(current_sublist)

    return sublists


def complete_ascending_list(lst):
    if not lst:
        return []

    complete_list = []
    for i in range(len(lst) - 1):
        complete_list.append(lst[i])
        if lst[i + 1] - lst[i] > 1:
            for j in range(lst[i] + 1, lst[i + 1]):
                complete_list.append(j)

    complete_list.append(lst[-1])

    return complete_list


if __name__ == "__main__":
    args = get_args()

    onnx_model = onnx.load(args.onnx_path)
    model_name = Path(args.onnx_path).stem

    initializers = [init.name for init in onnx_model.graph.initializer]
    nodes = onnx_model.graph.node

    graph_node_names = {}
    initializer_indexes = {}
    nodes_to_quantize = []

    pbar = tqdm(total=len(nodes), desc="Generate list node name of layers")

    initializer_index = 0

    for i, node in enumerate(nodes):
        graph_node_names[i] = node.name
        try:
            if len(node.input) > 0:
                intersection = list(filter(lambda x: x in node.input, initializers))
                if intersection:
                    nodes_to_quantize.append(node.name)
                    initializer_indexes[initializer_index] = {
                        "name": node.name,
                        "graph_index": i,
                    }
                    initializer_index += 1
            pbar.update(1)
        except Exception as e:
            print(f"ERROR: {e}")
            continue

    pbar.close()

    if args.only_weight:
        config_dir = f"configs/{model_name}/only_weight"
    else:
        config_dir = f"configs/{model_name}/ops_weight"

    os.makedirs(config_dir, exist_ok=True)

    k = int(len(nodes_to_quantize) * (1 - args.dropout))
    quantized_ratio = int((k / len(nodes_to_quantize)) * 100)

    # print(len(nodes_to_quantize), k, quantized_ratio)

    combinations_list = combinations(nodes_to_quantize, k)

    full_configurations = []
    limit = 1000

    i = 0
    while i < limit:
        combo = list(next(combinations_list))

        init_indexes = []
        for node_name in combo:
            index_of_init = [
                ind
                for ind, init in initializer_indexes.items()
                if init["name"] == node_name
            ][0]
            init_indexes.append(index_of_init)

        increasing_sub_indexes = split_increasing_sublists(init_indexes)
        # print(increasing_sub_indexes)

        for sub_indexes in increasing_sub_indexes:
            graph_indexes = [
                initializer_indexes[ind]["graph_index"] for ind in sub_indexes
            ]
            if not args.only_weight:
                graph_indexes = complete_ascending_list(graph_indexes)
            corresponding_node_names = [graph_node_names[ind] for ind in graph_indexes]
            # print(graph_indexes)
            # print(len((corresponding_node_names)))

            full_configurations.append(corresponding_node_names)

        # print("\n\n\n")

        i += 1

    config_path = os.path.join(config_dir, f"{model_name}_{quantized_ratio}.pkl")

    with open(config_path, "wb") as pickle_file:
        pickle.dump(full_configurations, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)

        print(
            f"Saved configure to {config_path} with {len(full_configurations)} scenarios"
        )
