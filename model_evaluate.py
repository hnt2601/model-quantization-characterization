import os
import time
import torch
import torchvision as tv
import onnxruntime
import numpy as np
import random

from tqdm import tqdm
from torch.utils.data import DataLoader
from common import MEAN, STD
from power import PowerConsumption


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    torch.cuda.manual_seed(worker_seed)
    torch.cuda.manual_seed_all(worker_seed)


def accuracy(true_labels, predicted_labels):
    correct_predictions = sum(
        1 for true, predicted in zip(true_labels, predicted_labels) if true == predicted
    )
    total_predictions = len(true_labels)
    accuracy = correct_predictions / total_predictions

    return accuracy


def evaluate(test_set, model_path, batch_size=1):
    comsp = PowerConsumption(device_id=0)

    seed = torch.initial_seed() % 2**32
    g = torch.Generator()
    g.manual_seed(seed)

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=os.cpu_count(),
        worker_init_fn=seed_worker,
        generator=g,
    )

    pbar = tqdm(total=len(test_loader))

    ort_inputs = {}
    true_labels = []
    predicted_labels = []
    run_time = []

    # Create an ONNX Runtime session with GPU as the execution provider
    options = onnxruntime.SessionOptions()
    options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    # options.log_severity_level = 3  # Set log verbosity to see GPU provider details
    t0 = time.time()

    sess = onnxruntime.InferenceSession(
        model_path,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        sess_options=options,
    )

    duration = time.time() - t0
    comsp.measure_power_usage(duration) # First measure for model loading

    # Check the input and output names and shapes of the model
    input_name = sess.get_inputs()[0].name
    # output_name = sess.get_outputs()[0].name
    # input_shape = sess.get_inputs()[0].shape
    # output_shape = sess.get_outputs()[0].shape
    # print(f"Input name: {input_name}, Input shape: {input_shape}")
    # print(f"Output name: {output_name}, Output shape: {output_shape}")


    for inputs, labels in test_loader:
        try:
            inp_list = [inp.numpy() for inp in inputs]
            inps = np.stack(inp_list, axis=0)

            ort_inputs.update({input_name: inps})
            
            t0 = time.time()
            
            predictions = sess.run(None, ort_inputs)[0]
            
            duration = time.time() - t0
            comsp.measure_power_usage(duration)

            for i in range(0, batch_size):
                pred_label = np.argmax(predictions[i], axis=0)
                true_label = labels[i].numpy()

                true_labels.append(true_label)
                predicted_labels.append(pred_label)

            pbar.update(1)
        except Exception as e:
            print(e)
            continue

    pbar.close()

    power, energy = comsp.get_consumption()

    acc = accuracy(true_labels, predicted_labels)

    del sess
    del comsp

    return acc, power, energy


if __name__ == "__main__":
    data_path = "/media/data/hoangnt/Projects/Datasets"
    model_path = "pretrained/mobilenetv2/mobilenetv2.onnx"
    batch_size = 1024

    transform = tv.transforms.Compose(
        [tv.transforms.ToTensor(), tv.transforms.Normalize(MEAN, STD)]
    )

    dataset = tv.datasets.CIFAR10(
        root=data_path,
        train=False,
        download=False,
        transform=transform,
    )

    i = 0
    while i < 1:
        accuracy = evaluate(dataset, model_path, batch_size)

        print(round(accuracy, 4))

        i += 1

# 19a: 0.8212
# 19: 0.8216
