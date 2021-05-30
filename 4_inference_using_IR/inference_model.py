# =============================================================================
# Created By     : Maulik Pandya
# Created Date   : May 3, 2021
# Python Version : 3.6
# =============================================================================
"""This script is to infer openvino optimized model"""
# =============================================================================
# Import Required Libraries
# =============================================================================
import copy
import sys
from PIL import Image
import numpy as np
import openvino
import time

from keras.datasets import mnist
from keras.utils.np_utils import to_categorical

# from keras.utils import to_categorical
(X_train, y_train), (X_test, y_test) = mnist.load_data()


try:
    from openvino import inference_engine as ie
    from openvino.inference_engine import IENetwork, IECore
except Exception as e:
    exception_type = type(e).__name__
    print("The following error happened while importing Python API module:\n[ {} ] {}".format(exception_type, e))
    sys.exit(1)

def pre_process_image(image, img_height=28):
    # Model input format
    n, c, h, w = [1, 1, img_height, img_height]
    im = Image.fromarray(np.uint8(image))
    processedImg = im.resize((h, w), resample=Image.BILINEAR)

    # Normalize to keep data between 0 - 1
    processedImg = np.array(processedImg)
    processedImg = processedImg.reshape((n, c, h, w))
    return processedImg


# Plugin initialization for specified device and load extensions library if specified.
plugin_dir = r'./../Models/3_openvino_IR_model/frozen_model.mapping'
model_xml = r'./../Models/3_openvino_IR_model/frozen_model.xml'
model_bin = r'./../Models/3_openvino_IR_model/frozen_model.bin'

ie = IECore()
print(ie.available_devices)
net = ie.read_network(model=model_xml, weights=model_bin)
# net.batch_size = 16

d_name = "HETERO:GPU,CPU"

layer_map = ie.query_network(network=net,device_name=d_name)
for layer in layer_map:
    print("{} : {} ".format(layer,layer_map[layer]))


exec_net = ie.load_network(network=net, device_name=d_name)

nq = exec_net.get_metric("OPTIMAL_NUMBER_OF_INFER_REQUESTS")
print(nq)
exec_net = ie.load_network(network=net, device_name=d_name, num_requests=nq)


assert len(net.inputs.keys()) == 1
assert len(net.outputs) == 1
input_blob = next(iter(net.inputs))
out_blob = next(iter(net.outputs))
del net

count=0

st = time.time()
pred_id =[]
for i in range(len(X_test)):
    processedImg = pre_process_image(X_test[i])
    res = exec_net.infer(inputs={input_blob: processedImg})
    # Access the results and get the index of the highest confidence score
    output_node_name = list(res.keys())[0]
    res = res[output_node_name]
    # Predicted class index.
    idx = np.argsort(res[0])[-1]
    pred_id.append(idx)

    count += (idx == y_test[i])

st1 = time.time()
print("Time : {}".format(st1-st))
print("Accuracy = {}".format(count/len(X_test)))
print(pred_id[:20])
print(y_test[:20])




