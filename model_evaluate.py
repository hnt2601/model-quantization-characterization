import os
import torch
import torchvision as tv
import onnxruntime
import numpy as np
import random

from tqdm import tqdm
from torch.utils.data import DataLoader


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


def evaluate(data_path, model_path, batch_size=1):
    g = torch.Generator()
    g.manual_seed(0)

    transform = tv.transforms.Compose(
        [tv.transforms.ToTensor(),
         tv.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))]
    )
    
    dataset = tv.datasets.CIFAR10(
        root=data_path,
        train=False,
        download=False,
        transform=transform,
    )
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, worker_init_fn=seed_worker, generator=g)
    pbar = tqdm(total=len(dataset))

    ort_inputs = {}
    true_labels = []
    predicted_labels = []

    # Create an ONNX Runtime session with GPU as the execution provider
    options = onnxruntime.SessionOptions()
    options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    # options.log_severity_level = 3  # Set log verbosity to see GPU provider details
    sess = onnxruntime.InferenceSession(
        model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"], sess_options=options
    )

    # Check the input and output names and shapes of the model
    input_name = sess.get_inputs()[0].name
    # output_name = sess.get_outputs()[0].name
    # input_shape = sess.get_inputs()[0].shape
    # output_shape = sess.get_outputs()[0].shape
    # print(f"Input name: {input_name}, Input shape: {input_shape}")
    # print(f"Output name: {output_name}, Output shape: {output_shape}")

    
    for inputs, labels in dataloader:
        try:
            inp_list = [inp.numpy() for inp in inputs]
            inps = np.stack(inp_list, axis=0)

            ort_inputs.update({input_name: inps})

            predictions = sess.run(None, ort_inputs)[0]

            for i in range(0, batch_size):
                pred_label = np.argmax(predictions[i], axis=0)
                true_label = labels[i].numpy()

                true_labels.append(true_label)
                predicted_labels.append(pred_label)

        except Exception as e:
            print(e)
            continue

    pbar.close()

    acc = accuracy(true_labels, predicted_labels)

    return acc


if __name__ == "__main__":
    import os
    import onnx
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    
    data_path = "/media/data/hoangnt/Projects/Datasets"
    fp32_model_path = "weights/0_resnet50_cifar10_qop_quant.onnx"
    fp32_model = onnx.load(fp32_model_path)
    fp32_acc = evaluate(data_path, fp32_model_path)
    
    print(fp32_acc)
