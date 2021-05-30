# =============================================================================
# Created By     : Maulik Pandya
# Created Date   : May 3, 2021
# Python Version : 3.6
# =============================================================================
"""Approach 2 ::"""
"""This script is to convert Keras trained HDF5 (.h5) model to frozen protobuf (.pb)"""
# =============================================================================
# Import Required Libraries
# =============================================================================
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D
from keras import backend as K
import tensorflow as tf
# tf.compat.v1.disable_v2_behavior()


def model_defination():
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPool2D(strides=2))
    model.add(Conv2D(filters=48, kernel_size=(5, 5), padding='valid', activation='relu'))
    model.add(MaxPool2D(strides=2))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(84, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    return model


def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        # Graph -> GraphDef ProtoBuf
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph


if __name__=="__main__":

    save_pb_dir = r'./../Models/2_tf_frozen_PB_model'
    model_fine_name = r'./../Models/1_keras_h5_model/best_model.h5'

    # model loading can be performed 2 ways
    # 1) model load using "load_model" keras api
    # model = load_model(model_fine_name)

    # 2) get model defination, load model weights
    model = model_defination()
    model.load_weights(model_fine_name)

    # get frozen graph
    frozen_graph = freeze_session(K.get_session(), output_names=[out.op.name for out in model.outputs])

    # save PB model to disk
    tf.train.write_graph(frozen_graph, save_pb_dir, "frozen_model_1.pb", as_text=False)

