#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
import os
from glob import glob 
import matplotlib.pyplot as plt
import random
from time import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, roc_auc_score
from keras import models, layers
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard

data = pd.read_pickle("./balancedData_shuffled")
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

import math
total = data.shape[0]
test_part = math.ceil(total / 5)
train_part = total - test_part

X,Y = procData(0, train_part)
test_X, test_Y = procData(train_part, total)

# Label processing
y_train = []
y_test = []
for i in Y:
    if i == 0:
        y_train.append(np.array([1,0]))
    elif i == 1:
        y_train.append(np.array([0,1]))

for j in test_Y:
    if i == 0:
        y_test.append(np.array([1,0]))
    elif i == 1:
        y_test.append(np.array([0,1]))
        
y_train = np.asarray(y_train)
y_test = np.asarray(y_test)

## Define model
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Flatten, BatchNormalization, Dropout, Conv2D, MaxPooling2D
from keras.optimizers import SGD, RMSprop
from keras.callbacks import EarlyStopping
from densenet121 import DenseNet

x_train = np.asarray(X) / 255.0
x_test = np.asarray(test_X) / 255.0
# y_train = np.asarray(Y)
# y_test = np.asarray(test_Y)

model = DenseNet(reduction=0.5, classes=2)

# weight_path="{}_weights.best.hdf5".format('model')
# sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
# earlystopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=16, verbose=0, mode='auto')
# checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_weights_only=True)
# tensorboard = TensorBoard(log_dir="./logs/{}".format(time()))

model.load_weights('densenet_weights.h5')

pred = model.predict(x_test)
y_test = test_Y
tp = 0
fp = 0
tn = 0
fn = 0
for i in range(len(y_test)):
    if pred[i][0] > pred[i][1]:
        y_pred = 0
    else:
        y_pred = 1
    
    if (y_pred == 1) and (y_test[i] == 1):
        tp = tp + 1
    elif (y_pred == 1) and (y_test[i] == 0):
        fp = fp + 1
    elif (y_pred == 0) and (y_test[i] == 0):
        tn = tn + 1
    elif (y_pred == 0) and (y_test[i] == 1):
        fn = fn + 1


acc = (tp + tn) / (tp + fp + tn + fn)

print(acc)
