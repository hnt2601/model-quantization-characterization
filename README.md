# TODO
## Table of Contents

- [TODO](#TODO)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Installation](#installation)
  - [Usage](#usage)
	  - [Command Line](#command-line)
  - [Examples](#examples)
  - [TODO](#todo)
## Introduction

This tool to 

This tool is designed to assess the reduction in accuracy when using INT8 models in comparison to FP32 models for node configuration combinations that will be quantized by the ONNX SDK. In doing so, we only quantize a certain proportion, denoted as 'x%' of nodes that contain parameters, such as Conv2D and FC layers.
The experiment is evaluated for accuracy on the CIFAR-10 dataset.

For example, we may choose to quantize 90% of the Conv2D and FC nodes in a ResNet50 model.


## Installation

You can easily install this package by the command
```bash
conda create -n <environment-name> --file requirements.txt
```
## Usage
### Command-line
Fine-tuing models
```bash
python cifar10_trainer.py
```

Convert Pytorch to onnx
```bash
python convert_to_onnx.py --pretrained path/to/file --backbone [model_name]
```

Generate permuation configures
```bash
python generate_combination_layers.py --onnx_path path/to/file
```

Model quantization and statistical analysis
```bash
python onnx_quantization.py --static --input_model /path/to/file --config_path /path/to/file --data_path /path/to/file --report_path /path/to/file --profiling
```

## TODO