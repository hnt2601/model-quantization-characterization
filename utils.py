import torch
import time
import openpyxl
import numpy as np
import onnxruntime
import typing
import os

from memory_profiler import profile


#@profile
def benchmark(model_path, batch_size=1, use_gpu=True):
    # Create an ONNX Runtime session with GPU as the execution provider
    options = onnxruntime.SessionOptions()
    options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.enable_profiling = True

    providers = ["CPUExecutionProvider"]
    if use_gpu and torch.cuda.is_available():
        providers.insert(0, "CUDAExecutionProvider")
        
    sess = onnxruntime.InferenceSession(
        model_path, providers=providers, sess_options=options
    )

    runs = 10
    device_name = "cpu" if not use_gpu else "cuda"
    device_index = 0
    run_time = []

    # Create an I/O binding
    io_binding = sess.io_binding()

    for i in range(runs):

        for in_tensor in sess.get_inputs():
            # Check the input and output names and shapes of the model
            input_name = in_tensor.name
            input_shape = [batch_size] + in_tensor.shape[1:]
            # print(f"Input name: {input_name}, Input shape: {input_shape}")

            # input_data = torch.empty(
            #     tuple(input_shape), device=device_name
            # ).contiguous()

            input_data = np.random.rand(*input_shape).astype(np.float32)

            X_ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(
                input_data, device_name, device_index
            )

            # Bind input data to the model
            io_binding.bind_input(
                name=input_name,
                device_type=device_name,
                device_id=device_index,
                element_type=input_data.dtype,
                shape=X_ortvalue.shape(),
                buffer_ptr=X_ortvalue.data_ptr(),
            )

        for out_tensor in sess.get_outputs():
            # Check the input and output names and shapes of the model
            output_name = out_tensor.name
            output_shape = [batch_size] + out_tensor.shape[1:]
            # print(f"Output name: {output_name}, Output shape: {output_shape}")

            # Bind output buffer

            # output_data = torch.empty(
            #     tuple(output_shape), device=device_name
            # ).contiguous()

            output_data = np.random.rand(*output_shape).astype(np.float32)

            Y_ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(
                output_data, device_name, device_index
            )

            io_binding.bind_output(
                name=output_name,
                device_type=device_name,
                device_id=device_index,
                element_type=output_data.dtype,
                shape=Y_ortvalue.shape(),
                buffer_ptr=Y_ortvalue.data_ptr(),
            )

        # Run inference
        start = time.time()
        
        sess.run_with_iobinding(io_binding)

        end = time.time() - start
        run_time.append(end)
        
    latency = round(np.mean(run_time) * 1000, 2)
    
    prof_file = sess.end_profiling()
    
    del sess
    del io_binding

    return latency, prof_file


# @profile
# def benchmark(model_path, batch_size=1, use_gpu=True):
#     # Create an ONNX Runtime session with GPU as the execution provider
#     options = onnxruntime.SessionOptions()
#     # options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
#     options.enable_profiling = True

#     providers = ['CPUExecutionProvider']
#     if use_gpu and torch.cuda.is_available():
#         providers.insert(0, "CUDAExecutionProvider")

#     sess = onnxruntime.InferenceSession(
#         model_path, providers=providers, sess_options=options
#     )
    
#     device_name = "cpu" if not use_gpu else "cuda"
#     device_index = 0

#     runs = 1
#     run_time = []

#     for i in range(runs):

#         inputs = {}

#         for in_tensor in sess.get_inputs():
#             # Check the input and output names and shapes of the model
#             input_name = in_tensor.name
#             input_shape = [batch_size] + in_tensor.shape[1:]
#             # print(f"Input name: {input_name}, Input shape: {input_shape}")

#             input_data = np.random.rand(*input_shape).astype(np.float32)

#             X_ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(
#                 input_data, device_name, device_index
#             )

#             inputs[input_name] = X_ortvalue

#         # Run inference
        
#         start = time.time()
        
#         # sess.run(None, inputs)
        
#         sess.run_with_ort_values(None, inputs)

#         end = time.time() - start
        
#         time.sleep(1)

#         run_time.append(end)

#     latency = np.median(run_time) * 1000

#     return latency


# def save_subgraphs_to_excel(model_path, subgraphs):
#     model_name = model_path.stem
#     # Create an Excel workbook if it doesn't exist
#     excel_file_path = subgraph_folder_path / "model_subgraphs.xlsx"
#     if not excel_file_path.exists():
#         workbook = openpyxl.Workbook()
#     else:
#         workbook = openpyxl.load_workbook(excel_file_path)
#         if "Sheet" in workbook.sheetnames:
#             workbook.remove(workbook["Sheet"])

#     # Check if the model sheet already exists
#     if model_name in workbook.sheetnames:
#         workbook.remove(workbook[model_name])

#     worksheet = workbook.create_sheet(title=model_name)

#     # Add headers to the sheet
#     worksheet["A1"] = "Subgraph"
#     worksheet["B1"] = "Length"
#     worksheet["C1"] = "Start Node"
#     worksheet["D1"] = "Nodes"
#     worksheet["E1"] = "Names"

#     # Iterate over the subgraphs and add them to the sheet
#     row_index = 2
#     subgraph_number = 0
#     for subgraph_set in subgraphs:
#         for subgraph in subgraph_set:
#             nodes = subgraph.nodes()
#             start_node_index = subgraph.indices[0]

#             worksheet.cell(
#                 row=row_index,
#                 column=1,
#                 value=f"{model_name}_subgraph_{subgraph_number}",
#             )
#             worksheet.cell(row=row_index, column=2, value=len(subgraph.nodes()))
#             worksheet.cell(row=row_index, column=3, value=start_node_index)

#             subgraph_nodes = ""
#             subgraph_names = ""

#             for j, node in enumerate(nodes):
#                 subgraph_nodes += str(node.op_type)
#                 subgraph_names += str(node.name)

#                 if j < len(nodes) - 1:
#                     subgraph_nodes += " _ "
#                     subgraph_names += " _ "

#             worksheet.cell(row=row_index, column=4, value=subgraph_nodes)
#             worksheet.cell(row=row_index, column=5, value=subgraph_names)

#             row_index += 1
#             subgraph_number += 1

#             # if start_node in pattern:

#     # Save the workbook
#     workbook.save(excel_file_path)
#     print(f"All subgraphs saved as: {excel_file_path}")
