print("loading packages;")
import pandas as pd
import numpy as np
from pickle import load, dump

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import roc_curve, auc, roc_auc_score
from keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

print("building model;")
def alexnet_model(img_shape=(50, 50, 3), n_classes=2, l2_reg=0.,
    weights=None):

    # Initialize model
    alexnet = Sequential()

    # Layer 1
    alexnet.add(Conv2D(96, (11, 11), input_shape=img_shape,
        padding='same', kernel_regularizer=l2(l2_reg)))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(MaxPooling2D(pool_size=(2, 2)))

    # Layer 2
    alexnet.add(Conv2D(256, (5, 5), padding='same'))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(MaxPooling2D(pool_size=(2, 2)))

    # Layer 3
    alexnet.add(ZeroPadding2D((1, 1)))
    alexnet.add(Conv2D(512, (3, 3), padding='same'))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(MaxPooling2D(pool_size=(2, 2)))

    # Layer 4
    alexnet.add(ZeroPadding2D((1, 1)))
    alexnet.add(Conv2D(1024, (3, 3), padding='same'))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))

    # Layer 5
    alexnet.add(ZeroPadding2D((1, 1)))
    alexnet.add(Conv2D(1024, (3, 3), padding='same'))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(MaxPooling2D(pool_size=(2, 2)))

    # Layer 6
    alexnet.add(Flatten())
    alexnet.add(Dense(3072))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(Dropout(0.5))

    # Layer 7
    alexnet.add(Dense(4096))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(Dropout(0.5))

    # Layer 8
    alexnet.add(Dense(n_classes))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('softmax'))

    if weights is not None:
        alexnet.load_weights(weights)

    return alexnet

def procData(data, lower_ind, upper_ind):
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
                y.append(np.asarray([0,1]))
            else:
                y.append(np.asarray([1,0]))
    
    return x, y
    
print("loading data;")
file_path = "/projectnb/cs542sp/idc_classification/balancedData_shuffled"
all_data = pd.read_pickle(file_path)
all_data.head()

X_red, y_red = procData(all_data, 0,100000)
X_red, y_red = np.asarray(X_red), np.asarray(y_red)

X_test, y_test = procData(all_data, 200000, 230000)
X_test, y_test = np.asarray(X_test), np.asarray(y_test)

epochs = 70
validation_split = .4
batch_size = 2048

model = alexnet_model()

weight_path="{}_weights.best.hdf5".format('model')
earlystopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=0, mode='auto')
checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_weights_only=True)

print("fitting model;")
model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
hist = model.fit(x = X_red, y = y_red, epochs=epochs, batch_size = batch_size, verbose = 2, callbacks = [earlystopping, checkpoint], validation_split=validation_split)
print("predicting test dats;")
test_loss,test_acc = model.evaluate(X_test, y_test, batch_size = batch_size)

print("saving models and weights;")
model.save("trainedmodel_GPU.h5") # saving the model 

print("saving h5 history;")
with open('trainHistoryOld', 'wb') as handle: # saving the history of the model trained for another 50 Epochs
    dump(hist.history, handle)

print("saving npy history")
history_acc_loss = {"loss": hist.history['loss'], "val_loss": hist.history['val_loss'], "acc": hist.history['acc'], "val_acc": hist.history['val_acc'], "final_test_loss_acc": [test_loss,test_acc]}

np.save("history_acc_loss.npy", history_acc_loss)

print("ploting")
# Plot training & validation acc values
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
plt.savefig("lost.png")
plt.close()

print("finished;")
