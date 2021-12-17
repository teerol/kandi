import time
import sys

import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt

# helper classes for saving structural data
class engine_data():
    def __init__(self,c,i,o,s,b) -> None:
        self.context = c
        self.inputs = i
        self.outputs = o
        self.stream = s
        self.bindings = b

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

def create_engine(engine_file):
    # Creates exutable trtengine form saved .trtengine -file

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    with open(engine_file, "rb") as f:
        serialized_engine = f.read()
    engine = runtime.deserialize_cuda_engine(serialized_engine)

    inputs, outputs, bindings, stream = allocate_buffers(engine)
    context = engine.create_execution_context()

    return engine_data(context,inputs,outputs,stream,bindings)

def allocate_buffers(engine):
    # helper function for create_engine
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream

def run_inference(in_data_preprocessed, engine_d):
    # runs inference on exutable trtengine
    # returns also minimalistic timing performce

    np.copyto(engine_d.inputs[0].host, in_data_preprocessed.ravel())
    s = time.time()
    [cuda.memcpy_htod_async(inp.device, inp.host, engine_d.stream) for inp in engine_d.inputs]
    engine_d.context.execute_async_v2(engine_d.bindings, engine_d.stream.handle)
    [cuda.memcpy_dtoh_async(out.host, out.device, engine_d.stream) for out in engine_d.outputs]

    engine_d.stream.synchronize()
    e = time.time()
    output = [out.host for out in engine_d.outputs]
    return output, e-s
