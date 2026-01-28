from keras.utils import plot_model
import tensorflow as tf
import numpy as np
import keras.layers as layers
from keras.models import Sequential, Model
from keras.applications import VGG16
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,  LearningRateScheduler
from keras.layers import Activation, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Conv2DTranspose
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import cv2
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l1, l2
l1_value = 0.001
l2_value = 0.001
model = Sequential()

# First Convolutional Layer
model.add(Conv2D(64, (3, 3), input_shape=(80, 80, 3), padding='same', kernel_regularizer=l1(l1_value), bias_regularizer=l2(l2_value)))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=l1(l1_value), bias_regularizer=l2(l2_value)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Second Convolutional Layer
model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Third Convolutional Layer
model.add(Conv2D(256, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(256, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(256, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Fourth Convolutional Layer
model.add(Conv2D(512, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(512, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(512, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Fifth Convolutional Layer
model.add(Conv2D(512, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(512, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(512, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Fully Connected Layers
model.add(Flatten())
model.add(Dense(4096, kernel_regularizer=l1(l1_value), bias_regularizer=l2(l2_value)))
model.add(Activation('relu'))
#model.add(Dropout(0.5))
model.add(Dense(4096, kernel_regularizer=l1(l1_value), bias_regularizer=l2(l2_value)))
model.add(Activation('relu'))
#model.add(Dropout(0.5))
model.add(Dense(2, kernel_regularizer=l1(l1_value), bias_regularizer=l2(l2_value)))  # Two output neurons for binary classification
model.add(Activation('softmax'))

plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)