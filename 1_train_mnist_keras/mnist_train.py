# =============================================================================
# Created By     : Maulik Pandya
# Created Date   : May 3, 2021
# Python Version : 3.6
# =============================================================================
"""The Script is to train CNN model to classify MNIST images using Keras"""
# =============================================================================
# Import Required Libraries
# =============================================================================

from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau
from numpy import random
import os
import numpy as np

# Enable/disable GPU as per system config
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Fixed seed is used to reproduce the same results
random.seed(123)

# download MNIST data and split into train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# reshape data to fit model
X_train = X_train.reshape(60000,28,28,1)
X_test = X_test.reshape(10000,28,28,1)

X_train = np.asarray(X_train)
X_test = np.asarray(X_test)
print(X_train.shape, X_test.shape)

# one-hot encode target column
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Model defination
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5,5), padding='same', activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPool2D(strides=2))
model.add(Conv2D(filters=48, kernel_size=(5,5), padding='valid', activation='relu'))
model.add(MaxPool2D(strides=2))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(84, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.build()
model.summary()

adam = Adam(lr=5e-4)
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=adam)

# Set a learning rate annealer
reduce_lr = ReduceLROnPlateau(monitor='val_acc', patience=3,
                                verbose=1, factor=0.2,min_lr=1e-6)
                                
# train the model
results = model.fit(X_train, y_train,batch_size=128, validation_data=(X_test, y_test), epochs=10,callbacks=[reduce_lr])



predTrain = model.evaluate(X_train,y_train)
pred = model.evaluate(X_test,y_test)

# save the model
model.save('./../Models/1_keras_h5_model/best_model.h5')
print('Model saved to disk')
