import urllib.request
import requests
import shutil
import json
import csv
import cv2
import scipy
import scipy.misc
import numpy as np
import pickle
import scipy.misc
import glob
import argparse

from keras import regularizers
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.layers import Input, Flatten, Dense
from keras.models import Model
from keras.optimizers import SGD, Adam
from keras.utils import to_categorical
from keras.applications.resnet50 import ResNet50, preprocess_input

from sklearn.model_selection import train_test_split

from CNN import CNN_model

parser = argparse.ArgumentParser()
parser.add_argument('--style', '-s', type=str, help='color or gray', default='color')
parser.add_argument('--batch_size', '-bs', type=int, help='batch size', default=32)
parser.add_argument('--learning_rate', '-lr', type=float, help='learning rate', default=0.007)
parser.add_argument('--epochs', '-ep', type=int, help='Epochs', default=60)
parser.add_argument('--model_name', '-mn', type=str, help='save model name', default='classifier.h5')
args = parser.parse_args()


newMeta = open('meta.txt','w')
for file in glob.glob('imgs/women/*.jpg'):
    newMeta.write('{},{}\n'.format(file, 0))
for file in glob.glob('imgs/men/*.jpg'):
    newMeta.write('{},{}\n'.format(file, 1))
for file in glob.glob('imgs/baby/*.jpg'):
    newMeta.write('{},{}\n'.format(file, 2))
newMeta.close()

newMeta = open('meta.txt', "rt")
X = []
Y = []

for line in newMeta:
    try :
        row = line.split(',')
        pic = scipy.misc.imread(row[0], mode ='RGB')
        X.append(np.array(pic/255.))
        if int(row[1].replace('\n','').strip()) == 0:
            Y.append(np.array([1,0,0]))

        elif int(row[1].replace('\n','').strip()) == 1:
            Y.append(np.array([0,1,0]))

        else:
            Y.append(np.array([0,0,1]))

    except Exception as e:
        print('Exception:'+str(e))

X = np.array(X)
Y = np.array(Y)



X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=729)


'''Model Definition'''

def create_base_model( w = 'imagenet', trainable = False):
    # model = VGG16(weights=w, include_top=False, input_shape=(224, 224, 3))
    model = ResNet50(weights=w, include_top=False, input_shape=(224, 224, 3))
    if(not trainable):
        for layer in model.layers:
            layer.trainable = False
    return model

def rebase_base_model(model):
    for layer in model.layers:
        layer.trainable = True
    return model

def add_custom_layers(base_model):
        x = base_model.output
        x = Flatten()(x)
        x = Dense(1024, activation='relu', kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l2(0.005))(x)
        x = Dense(512, activation='relu', kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l2(0.005))(x)
        # y = Dense(1, activation='sigmoid')(x)
        y = Dense(3, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=y)
        return model

model = create_base_model(w = 'imagenet', trainable = False)
model = add_custom_layers(model)
opt = Adam(lr=args.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
#In the summary, weights and layers from VGG part will be hidden, but they will be fit during the training
#model.summary()

try:
    history = model.fit(X_train, y_train, validation_data=[X_val, y_val], batch_size=args.batch_size, epochs=args.epochs)
    with open('history.pickle', 'wb') as handle:
        pickle.dump(history.history, handle, protocol=pickle.HIGHEST_PROTOCOL)
except Exception as e:
    print(e)
    model.save(args.model_name)


model.save(args.model_name)
