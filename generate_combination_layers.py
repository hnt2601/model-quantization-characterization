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

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_args()

    onnx_model = onnx.load(args.onnx_path)
    model_name = Path(args.onnx_path).stem
    
    initializers = [init.name for init in onnx_model.graph.initializer]
    nodes = onnx_model.graph.node
    
    nodes_to_quantize = []

    pbar = tqdm(total=len(nodes), desc="Generate list node name of layers")

    for node in nodes:
        try:
            if len(node.input) > 0:
                intersection = list(filter(lambda x: x in node.input, initializers))
                if intersection:
                    nodes_to_quantize.append(node.name)
            pbar.update(1)
        except Exception as e:
            print(f"ERROR: {e}")
            continue

    pbar.close()
    
    config_dir = f"configs/{model_name}"
    os.makedirs(config_dir, exist_ok=True)

    k = int(len(nodes_to_quantize) * (1-args.dropout))
    quantized_ratio = int((k/len(nodes_to_quantize))*100)
    
    print(len(nodes_to_quantize), k, quantized_ratio)
    
    combinations_list = combinations(nodes_to_quantize, k)
    
    config_list = []
    limit = 1000
    
    i = 0
    while i < limit:
        config_list.append(list(next(combinations_list)))
        
        i+=1
    
    config_path = os.path.join(config_dir, f"{model_name}_{quantized_ratio}.pkl")
    
    with open(config_path, 'wb') as pickle_file:
        pickle.dump(config_list, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"Saved configure to {config_path} with {len(config_list)} scenarios")