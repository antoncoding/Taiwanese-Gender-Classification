# python trainVG.py --batch_size=32 --epochs=30 -lr=0.01
import urllib.request
import requests
import shutil
#import time
#import datetime
import json
import csv
import cv2
import scipy
# import imageio
import scipy.misc
import numpy as np
import pickle
import scipy.misc
import glob
#import matplotlib.pyplot as plt

from urllib.parse import quote

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.layers import Input, Flatten, Dense
from keras.models import Model
from keras.optimizers import SGD, Adam
from keras import regularizers
import argparse

from keras.utils import to_categorical
# from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.resnet50 import ResNet50, preprocess_input

parser = argparse.ArgumentParser()
parser.add_argument('--style', '-s', type=str, help='color or gray', default='color')
parser.add_argument('--batch_size', '-bs', type=int, help='batch size', default=32)
parser.add_argument('--learning_rate', '-lr', type=float, help='learning rate', default=0.007)
parser.add_argument('--epochs', '-ep', type=int, help='Epochs', default=60)

args = parser.parse_args()


def urllib_get(uri):
    html_data = ''
    try:
        req = urllib.request.Request(uri)
        handler = urllib.request.urlopen(req);
        encoding = handler.headers.get_content_charset()
        html_data =  handler.read().decode(encoding)
    except Exception as e:
        print('Exception at urllib_get: '+str(e))
        pass
    return html_data

def DownloadFile(url, local_filename):
    try:
        r = requests.get(url)
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)
    except Exception as e:
        print("Exception, DownloadFile: "+str(e)+' : '+url)
        pass
    return

# data_list = LoadDataFromExcel('D:\Programming\Python\CNN\LINE_own_report.xlsx')
# pics = []
# file = open('train_list.txt','r')
# for line in file:
#     gender, url = line.split(',')[0], line.split(',')[1].split('\n')[0]
#     pics.append({'pic':url, 'gender':gender})


# file_reader= open('train_dataset_temp_2.csv', "rt", encoding='ascii')
# read = csv.reader(file_reader)
# X_train = []
# Y_train = []


# ''' Load X_train and Y_train from csv data '''
# for row in list(read)[1:]:
#     if args.style == 'color':
#         root_dir = 'color_face/'
#     elif args.style =='gray':
#         root_dir = 'gray_face/'

#     try :
#         fname = row[0][5:]
#         pic = imageio.imread(root_dir+fname)
#         X_train.append(np.array(pic/255.).reshape(224,224,-1))
#         if row[1].replace('\r','').replace('\n','').strip() == 'F':
#             Y_train.append(np.array(1))
#             # Y_train.append(np.array(1,0))
#         else:
#             Y_train.append(np.array(0))
#             # Y_train.append(np.array(0,1))
#     except Exception as e:
#         print('Exception:'+str(e))
#         pass

# X_train = np.array(X_train)
# Y_train = np.array(Y_train)


from sklearn.model_selection import train_test_split

#newMeta = open('meta.txt','w')
#for file in glob.glob('imgs/women/*.jpg'):
#    newMeta.write('{},{}\n'.format(file, 0))
#for file in glob.glob('imgs/men/*.jpg'):
#    newMeta.write('{},{}\n'.format(file, 1))
#for file in glob.glob('imgs/baby/*.jpg'):
#    newMeta.write('{},{}\n'.format(file, 2))
#newMeta.close()

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

#print(X_train.shape)



# Y_train = to_categorical(Y_train, num_classes=2)

# print(Y_train.shape)
# # print(X_train.shape)
# print("====================================>")

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
    model.save('gray_restnet.h5')


model.save('gray_restnet.h5')
