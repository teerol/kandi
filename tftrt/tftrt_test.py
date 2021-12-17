import time
import numpy as np
import tensorflow as tf

from test_bench_2 import preprocess_input, decode_predictions
from tensorflow.keras.utils import to_categorical

def create_inference_graph(input_saved_model):
  return tf.saved_model.load(input_saved_model)

def predict_tftrt(infer,ds):
  # evaluates model
  # Using TensorFlow inference
  top1 = 0
  top5 = 0
  total = 0
  for batch, yb in ds.as_numpy_iterator():
    batch = preprocess_input(batch)
    batch = tf.constant(batch)
    labeling = infer(batch)
    predictions = labeling.numpy()
    predictions = decode_predictions(predictions,top=5)
    ground_truth = decode_predictions(to_categorical(yb,1000), top=1)

    for i, img in enumerate(predictions): # predictions can be batch of multiple predictions
      total += 1
      print(total, end='\r')
      for j, pred in enumerate(img): # five predictions for each image
        if ground_truth[i][0][0] == pred[0]:
          top5 += 1
          if j == 0:
            top1 += 1
          break

  print("\ntop 1 accuracy: {:.2f}%\ntop 5 accuracy: {:.2f}%".format(top1/total*100,top5/total*100))


def benchmark_tftrt(infer, ds, N_run):
  # Performance test
  # Using TensorFlow inference

  N_warmup_run = 20
  times = np.empty((N_run,1))

  print(f"warm up for {N_warmup_run} rounds")
  for run in range(N_warmup_run):
    for batch, yb in ds.as_numpy_iterator():
      batch = preprocess_input(batch)
      batch = tf.constant(batch)
      labeling = infer(batch)

  nmb_photos=0
  for run in range(N_run):
    sum_times_run=0
    for batch, yb in ds.as_numpy_iterator():
      batch = preprocess_input(batch)
      nmb_photos += len(batch)
      batch = tf.constant(batch)
      start_time = time.time()
      labeling = infer(batch)
      end_time = time.time()
      sum_times_run += end_time-start_time
    times[run] = sum_times_run
    print("round {} took {:.2f}s".format(run+1,sum_times_run))

  print("TOOK {:.2f}s to predict {:.0f} photos for {} rounds"
        .format(np.sum(times),nmb_photos/N_run,N_run))
  print("On average {:.3f}s +- {:.3f}s per round"
        .format(np.average(times),np.std(times)))
  print("AND        {:.3f}ms +- {:.3f}ms per photo"
        .format(np.average(times/nmb_photos)*1000,np.std(times/nmb_photos)*1000))
  print("BEST FPS: {:.1f}".format(nmb_photos/N_run/np.min(times)))
  print("times:",times.T)
