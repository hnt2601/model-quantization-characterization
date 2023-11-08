from __future__ import print_function

import argparse
import io

import onnx
import torch

from backbone import *


def get_args():
    parser = argparse.ArgumentParser(description="Convert ONNX")
    parser.add_argument(
        "--input_size",
        default=[32, 32],
        nargs="+",
    )

    parser.add_argument(
        "--pretrained",
        type=str,
        default="./pretrained/model.pt",
    )

    parser.add_argument(
        "--backbone", choices=["resnet50", "mobilenetv2", "efficientnetv2"]
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_args()

    model_dict = {
        "resnet50": ResNet50(),
        "mobilenetv2": MobileNetV2(),
        "efficientnetv2": EfficientnetV2(),
    }

    model = model_dict[args.backbone]
    model.load_state_dict(torch.load(args.pretrained))
    model.eval()

    with torch.no_grad():
        print("Finished loading model!")

        # Export to ONNX
        input_names = ["input1"]
        output_names = ["output0"]
        input_shapes = {input_names[0]: [1, 3, *args.input_size]}
        onnx_bytes = io.BytesIO()
        zero_input = torch.zeros(*input_shapes[input_names[0]])

        dynamic_axes = {input_names[0]: {0: "batch"}}
        for _, name in enumerate(output_names):
            dynamic_axes[name] = dynamic_axes[input_names[0]]

        extra_args = {
            "opset_version": 13,
            "verbose": False,
            "input_names": input_names,
            "output_names": output_names,
            "dynamic_axes": dynamic_axes,
            "do_constant_folding": True,
            "export_params": True,
        }

        onnx_path = args.pretrained.replace(".pth", ".onnx")

        torch.onnx.export(model, zero_input, onnx_bytes, **extra_args)
        with open(onnx_path, "wb") as out:
            out.write(onnx_bytes.getvalue())

        onnx_model = onnx.load(onnx_path)
        inferred_model = onnx.shape_inference.infer_shapes(onnx_model)
        # print(onnx.helper.printable_graph(inferred_model.graph))
        onnx.save(inferred_model, onnx_path)

        print("Generated onnx model named {}".format(onnx_path))
