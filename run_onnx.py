# evaluate accuracy of NN-model saved in onnx-format.
# Using onnx-runtime

import onnx
import onnxruntime
import sys
import numpy as np

from test_bench_2 import load_data
from tensorflow.keras.utils import to_categorical

# works for any model that outputs bsx1000 array
from tensorflow.keras.applications.resnet50 import decode_predictions


def preprocess(img_data):
    # for each pixel in each channel, divide the value by 255
    # to get value between [0, 1] and then normalize
    mean_vec = np.array([0.485, 0.456, 0.406])
    stddev_vec = np.array([0.229, 0.224, 0.225])
    norm_img_data = np.zeros(img_data.shape).astype('float32')
    for i in range(img_data.shape[1]):
        norm_img_data[:,i,:,:] = (img_data[:,i,:,:]/255.0 - mean_vec[i]) / stddev_vec[i]
    return norm_img_data


def make_session(model_name):
    session = onnxruntime.InferenceSession(model_name, None)
    input_name = session.get_inputs()[0].name  
    return session,input_name


def main(model_file):
    ds = load_data(":", 1, False)
    total=0
    top1=0
    top5=0
    session,i_name = make_session(model_file)
    for batch, y in ds.as_numpy_iterator():
        # onnx-model takes input in (bs,3,224,224)
        batch = batch.transpose(0,3,1,2)
        b = preprocess(batch)
      
        out = session.run([], {i_name: b})
     
        out = out[0].reshape((1,1000))
        preds = decode_predictions(out, top=5)
        gt = decode_predictions(to_categorical(y,1000),top=1)
        
        for i, img in enumerate(preds):
            total += 1
        for j, pred in enumerate(img):
            if gt[i][0][0] == pred[0]:
                top5 += 1
                if j == 0:
                    top1 += 1
                break
        if total % 100 == 0:
            print("{} top : {:.2f}% top 5: {:.2f}%".format(total,top1/total*100,top5/total*100))
    print("\n")


if __name__ == '__main__':
    if len(sys.argv == 2):
        model_file = sys.argv[1]
        main(model_file)
    else:
        print("GIVE ONNX-MODEL FILE AS ARGUMENT")