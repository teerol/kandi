from tensorflow.python.compiler.tensorrt import trt_convert as trt

import sys
import os
import numpy as np
import tensorflow as tf

def main(name,precision):
    # converts saved model to TF-TRT

    gpu = tf.config.experimental.list_physical_devices('GPU')  

    # setting conversion parameters
    conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS
    conversion_params = conversion_params._replace(
            max_workspace_size_bytes=(20*10**8)) # 2MB
    conversion_params = conversion_params._replace(precision_mode=precision)
    conversion_params = conversion_params._replace(
            maximum_cached_engines=2)
    conversion_params = conversion_params._replace(
            minimum_segment_size=3)
    
            
    print(conversion_params)

    converter = trt.TrtGraphConverterV2(
        input_saved_model_dir=name,
        conversion_params=conversion_params)
    converter.convert()
    
    converter.build(input_fn=my_input_fn) # optional
    
    out_name = name.replace("/","_") + "TRT-TF_" + precision
    print(out_name, "CONVERTED")
    converter.save(out_name)
    return out_name

def my_input_fn():
    # Input for a single inference call
    Inp = np.random.normal(size=(32, 224, 224, 3)).astype(np.float32)
    yield (Inp,)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        name = str(sys.argv[1])
        assert os.path.exists(name)
        precision = str(sys.argv[2])
        assert precision.upper() in ['FP32','FP16']
        main(name, precision)
