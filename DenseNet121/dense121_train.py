#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
import os
from glob import glob
from time import time
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, roc_auc_score
from keras import models, layers
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

## Sort data with [path, id, label]
img_path = glob('/projectnb/cs542sp/idc_classification/data/*/*/*.png', recursive=True)
img_count = len(img_path)
img_name = [""] * img_count
img_label = [""] * img_count

data = pd.DataFrame(img_path, columns=['path'])
def extract_id(x):
    return x.split('/')[-1]
def extract_label(x):
    return x.split("class")[-1].split(".")[0]

data['id'] = data['path'].apply(extract_id)
data['label'] = data['id'].apply(extract_label)


## Read image to np array
def readImage(path):
    bgr_img = cv2.imread(path)
    b,g,r = cv2.split(bgr_img)
    rgb_img = cv2.merge([r,g,b])
    return rgb_img


## Oversampling by randomly flip images
def randomFlip(path):
    img = readImage(path)
    mode = random.randint(-1,1)
    return cv2.flip(img, mode)


## Read and oversample all image data
data = pd.read_pickle("./balancedData_shuffled")

def procData(lower_ind, upper_ind):
    x = []
    y = []
    for ind in range(lower_ind, upper_ind):
        path = data['path'][ind]
        label = data['label'][ind]
        image = data['matrix'][ind]
        shape = image.shape
        if shape == (50, 50, 3):
            x.append(image)
            if label == '1':
                y.append(1)
            else:
                y.append(0)

    return x, y


## Read full-load images
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
        y_train.append(np.array([1, 0]))
    elif i == 1:
        y_train.append(np.array([0, 1]))

for j in test_Y:
    if i == 0:
        y_test.append(np.array([1, 0]))
    elif i == 1:
        y_test.append(np.array([0, 1]))

y_train = np.asarray(y_train)
y_test = np.asarray(y_test)


## Define model
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, BatchNormalization, Dropout, Conv2D, MaxPooling2D
from keras.optimizers import SGD, RMSprop
from densenet121 import DenseNet

x_train = np.asarray(X) / 255.0
x_test = np.asarray(test_X) / 255.0
# y_train = np.asarray(Y)
# y_test = np.asarray(test_Y)

model = DenseNet(reduction=0.5, classes=2)

# model = Sequential()

#model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(50, 50, 3)))
#model.add(Conv2D(32, (3, 3), activation='relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))

#model.add(Conv2D(64, (3, 3), activation='relu'))
#model.add(Conv2D(64, (3, 3), activation='relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))

#model.add(Flatten())
#model.add(Dense(256, activation='relu'))
#model.add(Dropout(0.5))
#model.add(Dense(2, activation='softmax'))

weight_path="{}_weights.best.hdf5".format('model')
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
earlystopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=16, verbose=0, mode='auto')
checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_weights_only=True)
tensorboard = TensorBoard(log_dir="./logs/{}".format(time()))

model.compile(loss = 'binary_crossentropy', optimizer = sgd, metrics = ['accuracy'])

hist = model.fit(x_train, y_train, batch_size=128, epochs=50, callbacks = [earlystopping, checkpoint, tensorboard], validation_split = 0.3)
test_loss,test_acc = model.evaluate(x_test, y_test, batch_size=128) # original basize 32
# print(hist.history)


# Plot training & validation accuracy values

plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('Test accuracy : ' + str(test_acc))
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig("accuracy.png")
plt.close()

# Plot training & validation loss values
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Test loss : ' + str(test_loss))
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig("loss.png")

# Save model
model.save_weights('densenet_weights.h5')

print(score)
