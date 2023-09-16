#@ Importing packages
import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
import scipy
from PIL import Image
def load_signs_dataset():
    pass
def convert_to_one_hot():
    pass
import pandas as pd
import tensorflow as tf
import tensorflow.keras.layers as tfl
from tensorflow.python.framework import ops
from cnn_utils import *
from test_utils import summary, comparator
%matplotlib inline
np.random.seed(1)

# Preparing training and test set
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_happy_dataset()
# Normalize image vectors
X_train = X_train_orig/255.
X_test = X_test_orig/255.
# Reshape
Y_train = Y_train_orig.T
Y_test = Y_test_orig.T
#@ Sequential API CNN model to identify happy person             
def happyModel():                 
    model = tf.keras.Sequential([
            tf.keras.layers.ZeroPadding2D(padding=3, input_shape=(64,64,3)),                  # Padding of 3 to prevent shrink
            tf.keras.layers.Conv2D(filters=32, kernel_size= 7, strides=1),                    # Conv2D with 32 7x7 filters and stride of 1
            tf.keras.layers.BatchNormalization(axis=3),                                       # Batch normalization for 3r axis
            tf.keras.layers.ReLU(),                                                           # ReLU activation
            tf.keras.layers.MaxPool2D(),                                                      # pooling with default
            tf.keras.layers.Flatten(),                                                        # flattining the convolution
            tf.keras.layers.Dense(1, activation='sigmoid')                                    # Ouptut layer for prediction
        ])
    return model
happy_model = happyModel()                                                                    # calling model
happy_model.compile(optimizer='adam',                                                         # compiling model with adam
                   loss='binary_crossentropy',                                                # for sigmoid
                   metrics=['accuracy'])
happy_model.fit(X_train, Y_train, epochs=10, batch_size=16)                                   # training model 97% accuracy
happy_model.evaluate(X_test, Y_test)                                                          # testing with 94 % accuracy


# Loading the data (signs)
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_signs_dataset()

# Splitting into training and test sets
X_train = X_train_orig/255.
X_test = X_test_orig/255.
Y_train = convert_to_one_hot(Y_train_orig, 6).T
Y_test = convert_to_one_hot(Y_test_orig, 6).T
#@ Functional API CNN model to identify hand sign from image
def convolutional_model(input_shape):
    input_img = tf.keras.Input(shape=input_shape)
    Z1=tf.keras.layers.Conv2D(8,4,strides=1,padding='same')(input_img)         # CONV2D: 8 filters 4x4, stride of 1, padding 'SAME'
    A1=tf.keras.layers.ReLU()(Z1)                                              # RELU
    P1=tf.keras.layers.MaxPooling2D(8,strides=8,padding='same')(A1)            # MAXPOOL: window 8x8, stride 8, padding 'SAME'
    Z2=tf.keras.layers.Conv2D(16,kernel_size=2,strides=1,padding='same')(P1)   # CONV2D: 16 filters 2x2, stride 1, padding 'SAME'
    A2=tf.keras.layers.ReLU()(Z2)                                              # RELU
    P2=tf.keras.layers.MaxPooling2D(4,strides=4,padding='same')(A2)            # MAXPOOL: window 4x4, stride 4, padding 'SAME'
    F=tf.keras.layers.Flatten()(P2)                                            # FLATTEN
    outputs=tf.keras.layers.Dense(6, activation='softmax')(F)                  # ouptut layer for classfying sign 
    model = tf.keras.Model(inputs=input_img, outputs=outputs)
    return model
conv_model = convolutional_model((64, 64, 3))                                  # calling of CNN model made using functional api
conv_model.compile(optimizer='adam',                                           # compiling with adam 
                  loss='categorical_crossentropy',                             # for softmax
                  metrics=['accuracy'])
conv_model.summary()
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).batch(64)    # training
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test)).batch(64)       # testing
history = conv_model.fit(train_dataset, epochs=100, validation_data=test_dataset)   # running the model with training and testing set