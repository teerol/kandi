import numpy as np
import tensorflow as tf
import time

from tensorflow.keras.utils import to_categorical

from test_bench_2 import preprocess_input, decode_predictions

def make_tflite_interpreter(model_name, bs):
    interpreter = tf.lite.Interpreter(model_path=model_name)
    interpreter.resize_tensor_input(0, [bs,224,224,3])
    interpreter.allocate_tensors()
    input_indx = interpreter.get_input_details()[0]['index']
    output_indx = interpreter.get_output_details()[0]['index']
    return interpreter, input_indx, output_indx

def tf_lite_predict(interpreter, data, inx1, inxL):
    interpreter.set_tensor(inx1, data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(inxL)
    return output_data

def evaluate_tflite(tflite_model,ds,bs=32):
  global infer, s, e
  infer,s,e = make_tflite_interpreter(tflite_model, bs)
  top1 = 0
  top5 = 0
  total = 0
  for batch, yb in ds.as_numpy_iterator():
    batch = preprocess_input(batch)
    batch = tf.constant(batch)
    preds = tf_lite_predict(infer, batch, s, e)
    preds = decode_predictions(preds,top=5)
    gt = decode_predictions(to_categorical(yb,1000),top=1)

    for i, img in enumerate(preds):
      total += 1
      print(total, end='\r')
      for j, pred in enumerate(img):
        if gt[i][0][0] == pred[0]:
          top5 += 1
          if j == 0:
            top1 += 1
          break
        
  print("\ntop 1 accuracy: {:.2f}%\ntop 5 accuracy: {:.2f}%".format(top1/total*100,top5/total*100))


def benchmark_tflite(input_saved_model, ds, N_run):
    
    N_warmup_run = 2
    times = np.empty((N_run,1))

    for i in range(N_warmup_run):
      #print(f"WU {i+1}")
      for batch, yb in ds.as_numpy_iterator():
        batch = preprocess_input(batch)
        batch = tf.constant(batch)
        labeling = tf_lite_predict(infer, batch, s, e)
    nmb_photos=0
    for i in range(N_run):
      sum_run=0
      for batch, yb in ds.as_numpy_iterator():
        batch = preprocess_input(batch)
        nmb_photos += len(batch)
        batch = tf.constant(batch)
        start_time = time.time()
        labeling = tf_lite_predict(infer, batch, s, e)
        end_time = time.time()
        sum_run += end_time-start_time
      times[i] = sum_run
      print("round {} took {:.2f}s".format(i+1,sum_run))

    print("TOOK {:.2f}s to predict {:.0f} photos for {} rounds"
          .format(np.sum(times),nmb_photos/N_run,N_run))
    print("On average {:.3f}s +- {:.3f}s per round"
          .format(np.average(times),np.std(times)))
    print("AND        {:.3f}ms +- {:.3f}ms per photo"
          .format(np.average(times/nmb_photos)*1000,np.std(times/nmb_photos)*1000))
    print("times:",times.T)