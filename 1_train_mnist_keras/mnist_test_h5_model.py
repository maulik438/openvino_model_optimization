# =============================================================================
# Created By     : Maulik Pandya
# Created Date   : May 3, 2021
# Python Version : 3.6
# =============================================================================
"""This script is to test keras HDF5 model to classify MNIST images"""
# =============================================================================
# Import Required Libraries
# =============================================================================
import keras
from keras.utils.np_utils import to_categorical
from keras.datasets import mnist
from keras import models
import pydotplus as pydot
keras.utils.vis_utils.pydot = pydot
import numpy as np

# download MNIST data and split into train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# reshape data to fit model
X_train = X_train.reshape(60000,28,28,1)
X_test = X_test.reshape(10000,28,28,1)

X_train = np.asarray(X_train)
X_test = np.asarray(X_test)
print(X_train.shape, X_test.shape)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# load the model
model_path = r'./../Models/1_keras_h5_model/best_model.h5'
model = models.load_model(model_path)

# Evaluate model on train\test data
scoreTrain = model.evaluate(X_train, y_train, batch_size=32)
scoreTest = model.evaluate(X_test, y_test, batch_size=32)

# print train\test set results
print('*'*40)
print ("Train Loss = " + str(scoreTrain[0]))
print ("Train Accuracy = " + str(scoreTrain[1]))
print('*'*40)
print("Test Loss: %f" % scoreTest[0])
print("Test Accuracy: %f" % scoreTest[1])
print('*'*40)
