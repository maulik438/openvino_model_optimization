# =============================================================================
# Created By     : Maulik Pandya
# Created Date   : May 3, 2021
# Python Version : 3.6
# =============================================================================
"""This script is to infer MNIST data using PB model"""
# =============================================================================
# Import Required Libraries
# =============================================================================
import tensorflow as tf
from tensorflow.python.platform import gfile
from keras.datasets import mnist
import numpy as np

# download mnist data and split into train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# reshape data to fit model
X_train = X_train.reshape(60000,28,28,1)
X_test = X_test.reshape(10000,28,28,1)

X_train = np.asarray(X_train)
X_test = np.asarray(X_test)
print(X_train.shape, X_test.shape)

# load PB model
save_pb_dir = r'./../Models/2_tf_frozen_PB_model/frozen_model.pb'

f = gfile.FastGFile(save_pb_dir, 'rb')
graph_def = tf.GraphDef()
graph_def.ParseFromString(f.read())
f.close()

# initiate TF session
config = tf.ConfigProto(allow_soft_placement=True)
sess = tf.Session(config=config)
sess.graph.as_default()
tf.import_graph_def(graph_def)

# define input and output node name
# get node names from model.input/ model.output after loading .h5 model
input_node = 'import/dense_3/Softmax:0'
output_node = 'import/conv2d_1_input:0'

# verify model prediction for first 20 images
softmax_tensor = sess.graph.get_tensor_by_name(input_node)
predictions = sess.run(softmax_tensor, {output_node: X_test[:20]})
pred_num = np.argmax(predictions, axis=-1)

# print GT and predicted class for first 20 images
print(y_test[:20])
print(pred_num)
