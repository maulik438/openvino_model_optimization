# =============================================================================
# Created By     : Maulik Pandya
# Created Date   : May 3, 2021
# Python Version : 3.6
# =============================================================================
"""This script is to convert Keras trained HDF5 (.h5) model to frozen protobuf (.pb)"""
# =============================================================================
# Import Required Libraries
# =============================================================================
import tensorflow as tf
from tensorflow.python.framework import graph_io
from tensorflow.python.keras.models import load_model
tf.keras.backend.clear_session()


model_file_name = r'./../Models/1_keras_h5_model/best_model.h5'
save_pb_dir = r'./../Models/2_tf_frozen_PB_model'


def freeze_graph(graph, session, output, save_pb_dir='.', save_pb_name='frozen_model.pb', save_pb_as_text=False):
    with graph.as_default():
        graphdef_inf = tf.graph_util.remove_training_nodes(graph.as_graph_def())
        graphdef_frozen = tf.graph_util.convert_variables_to_constants(session, graphdef_inf, output)
        graph_io.write_graph(graphdef_frozen, save_pb_dir, save_pb_name, as_text=save_pb_as_text)
        return graphdef_frozen

# This line must be executed before loading Keras model.
tf.keras.backend.set_learning_phase(0)

# load model
model = load_model(model_file_name)

# Initialize session as per installed Tensorflow 1.0 / Tensorflow 2.0 version
session = tf.keras.backend.get_session()  # for TF 1.0
# session = tf.compat.v1.keras.backend.get_session()   # for TF 2.0

INPUT_NODE = [t.op.name for t in model.inputs]
OUTPUT_NODE = [t.op.name for t in model.outputs]
print(INPUT_NODE, OUTPUT_NODE)
frozen_graph = freeze_graph(session.graph, session, [out.op.name for out in model.outputs], save_pb_dir=save_pb_dir)
print('PB model is generated')
