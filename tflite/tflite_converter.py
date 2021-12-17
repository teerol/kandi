# saved model to tflite converter

import tensorflow as tf
import sys

def convert_to_tf_lite(model_dir, outname):
    converter = tf.lite.TFLiteConverter.from_saved_model(model_dir) # path to the SavedModel directory
    tflite_model = converter.convert()

    # Add possible optimizers
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    # Save the model.
    tflite_saved = "tfl/"+outname + ".tflite"
    with open(tflite_saved, 'wb') as f:
        f.write(tflite_model)
    print("Conversion done:",tflite_saved)

if __name__ == '__main__':
    if len(sys.argv) !=3:
        print("ERROR: GIVE SAVED MODEL DIR AND OUTNAME AS PARAMETER")
    else:
        convert_to_tf_lite(model_dir=sys.argv[1], outname=sys.argv[2])
