import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import sys
import argparse


from time import time
from tensorflow.keras.utils import to_categorical
from tensorflow.image import resize, resize_with_pad, resize_with_crop_or_pad


######### IMPORTANT! # change to correct model used to get right prerprocessing method. ###############
#from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input, decode_predictions
#from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
#from tensorflow.keras.applications.densenet import DenseNet121, preprocess_input, decode_predictions
### ##########################################################################################

if "tftrt" not in sys.modules.keys():
    import tftrt.tftrt_test as tftrt_test
    from trt.test_trt_engine import evaluate_trt, perf_test_trt

tf.autograph.set_verbosity(0)

def resize_img(img,lb):
  i = resize_with_crop_or_pad(img, 224,224) # apparently the best method
  #i = resize_with_pad(img, 224,224)
  #i = resize(img, [224,224])
  return (i,lb)


def data_generator(dataset):
    # yields the data in np.array
    # x.shape = (batch_size,224,224,3)
    # y.shape = (batch_size,1000)
    for batch in dataset.as_numpy_iterator():
        x = preprocess_input(batch[0])
        y = to_categorical(batch[1],1000)
        yield x,y


def load_data(split_a=":1%", batch_size=1, ret_gen=True):
    # loads data from tensorflow imagenet_v2 dataset

    dataset = tfds.load('imagenet_v2',
                split='test['+split_a+']',
                as_supervised=True)

    dataset = dataset.map(resize_img, num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False)
    dataset = dataset.batch(batch_size)
    if ret_gen:
      gen = data_generator(dataset)
      return gen
    else:
      return dataset

def make_model(model_name, save=True):
    # loads model from memory or downloads a new model
    try:
        model = tf.keras.models.load_model(model_name)
    except:
        model = ResNet50(weights='imagenet')#,alpha=1.0) #alpha for mobilenets
        model.trainable=False
        model.compile(optimizer='adam',
                loss=tf.keras.losses.CategoricalCrossentropy(), 
                metrics=['accuracy','top_k_categorical_accuracy'])
        if save:
            model.save("models/"+model_name)
    return model


def evaluate_model(model, x_test):
    # evaluates model top-1 and top-5 five accuracies using x_test as test data.
    # Using Keras-interface.
    # for optimal memory consumption use genarator for x_test.
    history = model.evaluate(x_test)   
    top1_acc = history[1]
    top5_acc = history[2]
    print("top 1 accuracy: {:.2f}%".format(top1_acc*100))
    print("top 5 accuracy: {:.2f}%".format(top5_acc*100))
    

def perf_test_model(model,x,rounds,bs):
    # Makes performance test for model using data x and batch size bs
    # Using Keras interface

    N_warm_up = 20
    #with tf.device('/CPU:0'): # use for only CPU-tests
    
    x_stable = x
    
    nmb_photos = len(x_stable)*bs
    print(f"perftesting about {nmb_photos} photos for {rounds} times (batch_size={bs})")
    # warmup
    print(f"warm up for {N_warm_up} rounds")
    for w in range(N_warm_up):
        model.predict(x_stable)

    times = np.empty((rounds,1))

    for i in range(rounds):
        s = time()
        model.predict(x_stable)
        e = time()
        times[i] = e-s
        print("round {} took {:.2f}s".format(i+1,e-s))

    print("TOOK {:.2f}s to predict {} photos for {} rounds"
        .format(np.sum(times),nmb_photos,rounds))
    print("On average {:.3f}s +- {:.3f}s per round"
        .format(np.average(times),np.std(times)))
    print("AND        {:.3f}ms +- {:.3f}ms per photo"
        .format(np.average(times/nmb_photos)*1000,np.std(times/nmb_photos)*1000))
    print("times:",times.T)
    print("BEST FPS: {:.1f}".format(nmb_photos/np.min(times)))


def main(data_split, batch_size, eval, perf, model_name, tf_trt, trt):
    if tf_trt:
        infer = tftrt_test.create_inference_graph(model_name)
    elif trt:
        # due to some pycuda memory resource and runtime errors
        # trt engines are made in actual testing funtions
        pass
    else:
        model = make_model(model_name, save=True)
    print("Using model:",model_name)

    data_set = load_data(data_split, batch_size, ret_gen=False)
    if eval:
        if tf_trt:
            tftrt_test.predict_tftrt(infer, data_set)
        elif trt:
            evaluate_trt(data_set, model_name)
        else:
            data_gen = load_data(data_split, batch_size)
            evaluate_model(model, data_gen)
    if perf is not None:
        if tf_trt:
            tftrt_test.benchmark_tftrt(infer, data_set, perf)
        elif trt:
            perf_test_trt(data_set, model_name, perf)
        else:
            perf_test_model(model, data_set, perf, batch_size)

def parse_sysargs(args):
    parser = argparse.ArgumentParser(description='Test bench for Neural Network models and platforms')
    parser.add_argument('-e','--eval', action='store_true', help='Evaluate model with data')
    parser.add_argument('-p','--perf', metavar='rounds', type=int, help="Make performance tests (how many times to test the data)")
    parser.add_argument('-ds','--data_split', default='1%',type=str,help="used data split in tfds format e.x. 0:10 50percent 500: etc.")
    parser.add_argument('-bs','--batch_size', default=1,type=int,help="used batch_size during evaluation and performace tests")
    parser.add_argument('-m','--model', default="Plain_ResNet50",type=str,help="model used in tests can be tensorflow saved model, or .trtengine file")
    ns = parser.parse_args(args)
    return ns

if __name__ == '__main__':
    if len(sys.argv) <=1:
        sys.argv.append("-h") # print help if no arguments given
    arg = parse_sysargs(sys.argv[1:])

    trt_tests = False
    tf_trt_tests = False # change to True to run TF-TRT models on plain TensorFlow
    if arg.model.find("TRT-TF") != -1:
        tf_trt_tests = True
    if arg.model.find(".trtengine") != -1:
        trt_tests = True
    main(arg.data_split, arg.batch_size, arg.eval, arg.perf, arg.model, tf_trt_tests, trt_tests)