# neccesary libraries
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, Input
from keras.layers import (Dense, Dropout,concatenate, Activation, Flatten, BatchNormalization, Conv2D, MaxPool2D, MaxPooling2D,Input,GlobalAveragePooling2D,
GlobalMaxPooling2D,ZeroPadding2D,AveragePooling2D,Reshape,Convolution2D)
from keras.utils.np_utils import to_categorical
from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from keras.utils import layer_utils
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras.utils.vis_utils import plot_model

from sklearn.utils import class_weight
from sklearn import model_selection

from glob import glob
from shutil import copyfile
import pandas as pd
from os import listdir
import fnmatch
import numpy as np
import random
import cv2

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.utils import class_weight
from PIL import Image
from pandas import DataFrame

from time import time

# read data
data=pd.read_pickle("./balancedData_shuffled")

## get image data from dataframe
def procData(lower_ind, upper_ind):
    x = []
    y = []
    for ind in range(lower_ind, upper_ind):
        path = data['path'][ind]
        label = data['label'][ind]
        image = data['matrix'][ind]
        shape = image.shape
        if shape == (50,50,3):
            x.append(image)
            if label == '1':
                y.append(1)
            else:
                y.append(0)
    return x, y

## Read images for debugging
import math
total = data.shape[0]
test_part = math.ceil(total / 5)
train_part = total - test_part

X,Y = procData(0, train_part)
test_X, test_Y = procData(train_part, total)

print("idc(+) :", Y.count(1))
print("idc(-) :", Y.count(0))
print("Testing size :", len(test_Y))
print("Training data shape :", X[0].shape)
df = pd.DataFrame()
df["images"]=X
df["labels"]=Y
X2 = df["images"]
Y2 = df["labels"]
X = np.array(X)
Y = np.array(Y)

test_X = np.array(test_X)
test_Y = np.array(test_Y)

train_X = X/255.0 # scale to [0,1]
test_X = test_X/255.0 # scale to [0,1]

trainHot_Y = to_categorical(Y, num_classes = 2)
testHot_Y = to_categorical(test_Y, num_classes = 2)

print("vgg16 executing " )
def vgg_like(train_X, trainHot_Y, test_X, testHot_Y):
    
    input_shape = (50,50,3)
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(3, 3),activation='relu',input_shape=input_shape,strides=2))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    
    model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adadelta(lr=0.1),metrics=['accuracy'])
    
    es= EarlyStopping(monitor='val_loss', min_delta=0.1, patience=5, verbose=0, mode='auto')
    hist = model.fit(train_X,trainHot_Y, batch_size=64, epochs=20, callbacks = [es], validation_split =0.2)

    score = model.evaluate(test_X, testHot_Y, batch_size=128) # original basize 32
    
#     print(hist.history)

    plt.plot(hist.history['acc'])
    plt.plot(hist.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig("accuracy.png")
    plt.close()

    # Plot training & validation loss values
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig("loss.png")

    # Save model
    model.save_weights('xxx.h5')
    return model
vgg_like(train_X, trainHot_Y, test_X, testHot_Y)
