import numpy as np
import time

from tensorflow.python.eager.context import context
from trt.trt_inference import run_inference, create_engine
from test_bench_2 import decode_predictions, load_data
from tensorflow.keras.utils import to_categorical


def preprocess_img(img_data):
    mean_vec = np.array([0.485, 0.456, 0.406])
    stddev_vec = np.array([0.229, 0.224, 0.225])
    norm_img_data = np.zeros(img_data.shape).astype('float32')
    for i in range(img_data.shape[1]):
        # for each pixel in each channel, divide the value by 255 to get value between [0, 1] and then normalize
        norm_img_data[:,i,:,:] = (img_data[:,i,:,:]/255.0 - mean_vec[i]) / stddev_vec[i]
    return norm_img_data

def evaluate_trt(ds, engine_file):
    # Using TensorRT Python API
    # and onnx model file
    engine = create_engine(engine_file)

    total=0
    top1=0
    top5=0

    for batch, y in ds.as_numpy_iterator():
        # onnx-model takes input in (bs,3,224,224)
        batch = batch.transpose(0,3,1,2)
        b = preprocess_img(batch)
        
        out,_ = run_inference(b, engine)
        
        out = np.array(out).reshape((1,1000))
        preds = decode_predictions(out,top=5)
        ground_truth = decode_predictions(to_categorical(y,1000),top=1)
        
        for i, img in enumerate(preds):
            total += 1
            for j, pred in enumerate(img):
                if ground_truth[i][0][0] == pred[0]:
                    top5 += 1
                    if j == 0:
                        top1 += 1
                    break

        if total % 100 == 0:
            print("{} top : {:.2f}% top 5: {:.2f}%".format(total,top1/total*100,top5/total*100),end="\r")
    print("\ntop 1 accuracy: {:.2f}%\ntop 5 accuracy: {:.2f}%".format(top1/total*100,top5/total*100))

def perf_test_trt(ds, engine_file, N_run):
    # Using TensorRT Python API
    # and onnx model file

    engine = create_engine(engine_file)

    N_warmup_run = 20
    times = np.empty((N_run,1))

    print(f"warm up for {N_warmup_run} rounds")
    for i in range(N_warmup_run):
        for batch, yb in ds.as_numpy_iterator():
            batch = batch.transpose(0,3,1,2)
            batch = preprocess_img(batch)
            out,t = run_inference(batch, engine)

    nmb_photos=0
    for i in range(N_run):
        sum_run=0
        for batch, yb in ds.as_numpy_iterator():
            batch = batch.transpose(0,3,1,2)
            batch = preprocess_img(batch)
            nmb_photos += len(batch)
            start_time = time.time()
            out,t = run_inference(batch, engine)
            end_time = time.time()
            sum_run += t
        times[i] = sum_run
        print("round {} took {:.2f}s".format(i+1,sum_run))
    print("TOOK {:.2f}s to predict {:.0f} photos for {} rounds"
          .format(np.sum(times),nmb_photos/N_run,N_run))
    print("On average {:.3f}s +- {:.3f}s per round"
          .format(np.average(times),np.std(times)))
    print("AND        {:.3f}ms +- {:.3f}ms per photo"
          .format(np.average(times/nmb_photos)*1000,np.std(times/nmb_photos)*1000))
    print("BEST FPS: {:.1f}".format(nmb_photos/N_run/np.min(times)))
    print("times:",times.T)
