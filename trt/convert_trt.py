import sys
import os

import tensorrt as trt

def convert(onnx_file, precision, outname):
    # converts the onnx-model to TensorRT engine
    # Using TensorRT Python API

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    builder.max_batch_size = 1
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    success = parser.parse_from_file(onnx_file)
    for idx in range(parser.num_errors):
        print(parser.get_error(idx))

    if not success:
        print("404: SOMETHING WENT WRONG")
    config = builder.create_builder_config()
    if precision.upper() == "FP16":
        config.flags = 1<<int(trt.BuilderFlag.FP16)
    config.max_workspace_size = 1 << 20 # 1 MiB

    serialized_engine = builder.build_serialized_network(network, config)
    with open(outname+".trtengine", "wb") as f:
        f.write(serialized_engine)

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("GIVE: onnx file path _ precision _ out name")
    else:
        onnx_file = sys.argv[1]
        outname = sys.argv[2]
        precision = sys.argv[3]
        assert os.path.exists(onnx_file)
        convert(onnx_file, precision, outname)