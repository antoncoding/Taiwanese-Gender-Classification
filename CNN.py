import numpy as np

from keras.models import Sequential, load_model
from keras.constraints import maxnorm
from keras.layers import Dense, Dropout, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import SGD, Adam, Adagrad
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import backend as K
from keras.layers.normalization import BatchNormalization

def CNN_model():
    # create model
    model = Sequential()
    model.add(Conv2D(128, (3, 3), input_shape=(224,224,3), activation='relu',padding='same'))
    model.add(Dropout(0.2))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Dropout(0.3))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2)))

    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Dropout(0.35))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2)))
    model.add(Conv2D(1024, (3, 3), activation='relu', padding='same'))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dropout(0.45))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.45))
    model.add(Dense(3, activation='softmax'))

    # Compile model
    return model
