python subraph_quantization_benchmark.py --static --input_model pretrained/mobilenetv2/mobilenetv2.onnx \
                            --config_path configs/mobilenetv2/only_weight/mobilenetv2_79.pkl \
                            --data_path ../Datasets/cifar10/ --report_path reports/mobilenetv2_only_weight2.xlsx --profiling --benchmark