import argparse
import json
import torch
import random
import torchvision as tv
import openpyxl
import shutil
import numpy as np
import pickle

from model_evaluate import evaluate
from onnxruntime.quantization import (
    QuantFormat,
    QuantType,
    quantize_dynamic,
    quantize_static,
)
from torch.utils.data import DataLoader
from data import vision_data_reader
from utils import benchmark


def extract_analytic_to_excel(workbook, data, title):
    worksheet = workbook.create_sheet(title=title)

    # Add headers to the sheet
    worksheet["A1"] = "Metric"
    worksheet["B1"] = "FP32"
    worksheet["C1"] = "INT8"

    row_index = 2

    for metric, perf in data.items():
        worksheet.cell(row=row_index, column=1, value=metric)

        worksheet.cell(row=row_index, column=2, value=perf["FP32"])

        worksheet.cell(row=row_index, column=3, value=perf["INT8"])

        row_index += 1


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_model",
        default="./weights/resnet50_cifar10.onnx",
        help="Path to floating-point 32 bits model.",
    )
    parser.add_argument(
        "--config_path",
        default="config.pkl",
        help="Specify the destination folder of input data sets.",
    )
    parser.add_argument(
        "--quant_format",
        default=QuantFormat.QOperator,
        type=QuantFormat.from_string,
        choices=list(QuantFormat),
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/media/data/hoangnt66/Projects/Datasets",
    )
    parser.add_argument(
        "--report_path",
        type=str,
        default="reports/quantization_report.xlsx",
    )
    parser.add_argument(
        '--proc',
        type=int,
        default=1
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=256
    )

    parser.add_argument("--profiling", action="store_true", default=False, help="")
    parser.add_argument("--static", action="store_true", default=False, help="")
    parser.add_argument("--dynamic", action="store_true", default=False, help="")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    
    workbook = openpyxl.Workbook()
    
    fp32_model_path = args.input_model
    fp32_prof_file = fp32_model_path.replace("onnx", "json")
    
    print("benchmarking fp32 model...")
    fp32_latency, prof_file = benchmark(fp32_model_path, args.batch_size)
    fp32_latency /= args.batch_size

    if args.profiling:
        shutil.move(prof_file, fp32_prof_file)
    
    fp32_acc = evaluate(args.data_path, fp32_model_path, args.batch_size)
    # print(f"Avg: {fp32_latency:.2f}ms")
    
    mean, std = (0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)
    
    transform = tv.transforms.Compose(
        [tv.transforms.ToTensor(), tv.transforms.Normalize(mean, std)]
    )

    test_set = tv.datasets.CIFAR10(
        root=args.data_path,
        train=False,
        download=False,
        transform=transform,
    )
    
    nb_sample_to_calib = 1000
    random.shuffle(test_set.targets)
    filtered_indices = test_set.targets[:nb_sample_to_calib]
    calib_set = torch.utils.data.Subset(test_set, filtered_indices)
    calib_loader = DataLoader(calib_set, batch_size=args.batch_size, shuffle=False, drop_last=True)
    
    with open(args.config_path, 'rb') as pickle_load:
        quantize_config = pickle.load(pickle_load)
    
    ratio_quantized = int(args.config_path.split('.')[0].split('_')[-1])
    nb_index_per_proc = len(quantize_config) // args.proc
    start_index = 0
    end_index = start_index + nb_index_per_proc
    
    for i in range(start_index, end_index):
        
        nodes_to_quantize = quantize_config[i]
        
        int8_model_path = fp32_model_path.replace(".onnx", f"_quant_{i}.onnx")
        int8_prof_file = int8_model_path.replace("onnx", "json")

        if args.static:
            
            calibration_data_reader = vision_data_reader.VisionDataReader(
                calib_loader, fp32_model_path
            )
                    
            quantize_static(
                model_input=fp32_model_path,
                model_output=int8_model_path,
                calibration_data_reader=calibration_data_reader,
                quant_format=args.quant_format,
                per_channel=False,
                weight_type=QuantType.QInt8,
                activation_type=QuantType.QUInt8,
                nodes_to_quantize=nodes_to_quantize,
            )
            print("Calibrated and Static quantized model saved.")
            
        elif args.dynamic:
            quantize_dynamic(
                fp32_model_path,
                int8_model_path,
                weight_type=QuantType.QUInt8,
                per_channel=False,
            )
            print("Dynamic quantized model saved.")

        print("benchmarking int8 model...")
    
        int8_latency, prof_file = benchmark(int8_model_path)
        int8_latency /= args.batch_size
        
        if args.profiling:
            shutil.move(prof_file, int8_prof_file)
        
        int8_acc = evaluate(args.data_path, int8_model_path, args.batch_size)
        # print(f"Avg: {int8_latency:.2f}ms")
        
        data = {
            "Ratio Quantized Layer (%)": {"FP32": 0, "INT8": ratio_quantized},
            "Latency (s)": {"FP32": round(fp32_latency, 2), "INT8": round(int8_latency, 2)},
            "Latency Drop (%)": {"FP32": 0.0, "INT8": round(fp32_latency / int8_latency, 2)},
            "Accuracy": {"FP32": round(fp32_acc, 4), "INT8": round(int8_acc, 4)},
            "Accuracy Loss (%)": {"FP32": 0.0, "INT8": round(fp32_acc / int8_acc, 4)},
        }

        extract_analytic_to_excel(workbook, data, str(i))

    workbook.save(args.report_path)

if __name__ == "__main__":
    main()
