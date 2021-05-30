# =============================================================================
# Created By     : Maulik Pandya
# Created Date   : May 3, 2021
# Python Version : 3.6
# =============================================================================
"""This script is to infer openvino optimized model"""
# =============================================================================
# Import Required Libraries
# =============================================================================
import sys
import numpy as np
import openvino
import time

from keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()

try:
    from openvino import inference_engine as ie
    from openvino.inference_engine import IENetwork, IECore
except Exception as e:
    exception_type = type(e).__name__
    print("The following error happened while importing Python API module:\n[ {} ] {}".format(exception_type, e))
    sys.exit(1)


def pre_process_images(image,nq, img_height=28):
    # Model input format
    n, c, h, w = [nq, 1, img_height, img_height]
    image = image.reshape((n, c, h, w))
    return  image


# Plugin initialization for specified device and load extensions library if specified.
plugin_dir = r'./../Models/3_openvino_IR_model/frozen_model.mapping'
model_xml = r'./../Models/3_openvino_IR_model/frozen_model.xml'
model_bin = r'./../Models/3_openvino_IR_model/frozen_model.bin'

ie = IECore()
print(ie.available_devices)
net = ie.read_network(model=model_xml, weights=model_bin)
BATCH = 8
net.batch_size = BATCH

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

batch_count = np.int(np.ceil(len(X_test) /BATCH))
for i in range(batch_count):
    processedImg = pre_process_images(X_test[i*BATCH:(i+1)*BATCH],BATCH)
    res = exec_net.infer(inputs={input_blob: processedImg})
    # Access the results and get the index of the highest confidence score
    output_node_name = list(res.keys())[0]
    res = res[output_node_name]
    # Predicted class index.
    idx = np.argmax(res,axis=1)
    pred_id.extend(idx)

    for id,p in zip(idx,y_test[i*BATCH:(i+1)*BATCH]):
        count += (id == p)

st1 = time.time()
print("Time : {}".format(st1-st))
print("Accuracy = {}".format(count/len(X_test)))
print(pred_id[:20])
print(y_test[:20])




