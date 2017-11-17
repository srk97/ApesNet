from __future__ import absolute_import
from __future__ import print_function
import os

os.environ['KERAS_BACKEND'] = 'tensorflow'

import pylab as pl
import matplotlib.cm as cm
import itertools
from keras import layers
from keras.layers.core import Activation, Reshape
from keras.layers import convolutional,pooling,merge
from keras.models import Model
from keras.layers import Input,normalization,core
from keras.utils.visualize_util import plot
from keras.optimizers import SGD

import cv2
import numpy as np

path = './CamVid/'
#data_shape = 360*480
print (os.getcwd())

lRelu = layers.advanced_activations.LeakyReLU(alpha=0.3)


def normalized(rgb):
    # return rgb/255.0
    norm = np.zeros((rgb.shape[0], rgb.shape[1], 3), np.float32)

    b = rgb[:, :, 0]
    g = rgb[:, :, 1]
    r = rgb[:, :, 2]

    norm[:, :, 0] = cv2.equalizeHist(b)
    norm[:, :, 1] = cv2.equalizeHist(g)
    norm[:, :, 2] = cv2.equalizeHist(r)

    return norm


def one_hot_encode(labels):
    x = np.zeros([360, 480, 12])
    for i in range(360):
        for j in range(480):
            x[i, j, labels[i][j]] = 1
    return x


def prep_training_data():
    with open('CamVid/train.txt') as f:
        txt = f.readlines()
        txt = [line.split(' ') for line in txt]

    train_data = []
    train_labels = []
    for i in range(len(txt)):
        train_data.append(normalized(cv2.imread(os.getcwd() + txt[i][0][7:])))
        train_labels.append(one_hot_encode(cv2.imread(os.getcwd() + txt[i][1][7:][:-1])[:, :, 0]))

    return np.array(train_data), np.array(train_labels)


train_data, train_labels = prep_training_data()

train_labels = np.reshape(train_labels, (-1, 360*480, 12))

train_data = train_data[0:1]
train_labels = train_labels[0:1]

class_weighting= [0.2595, 0.1826, 4.5640, 0.1417, 0.5051, 0.3826, 9.6446, 1.8418, 6.6823, 6.2478, 3.0, 7.3614]

#ConvBlock1 (16,7,0.5)

inputImage = Input(shape=(360,480,3))
convBlock1 = convolutional.Convolution2D(16,7,7,border_mode='same')(inputImage)
convBlock1 = normalization.BatchNormalization(epsilon=1e-3)(convBlock1)
convBlock1 = core.Activation(lRelu)(convBlock1)
convBlock1 = core.Dropout(0.5)(convBlock1)
convBlock1 = pooling.MaxPooling2D(pool_size=(2,2),dim_ordering='tf')(convBlock1)

#end of ConvBlock1

#ConvBlock2 (64,7,0.5)

convBlock2 = convolutional.Convolution2D(64,7,7,border_mode='same')(convBlock1)
convBlock2 = normalization.BatchNormalization(epsilon=1e-3)(convBlock2)
convBlock2 = core.Activation(lRelu)(convBlock2)
convBlock2 = core.Dropout(0.5)(convBlock2)
convBlock2 = pooling.MaxPooling2D(pool_size=(2,2),dim_ordering='tf')(convBlock2)

#end of ConvBlock2

#ConvBlock3 (64,7,0.5)

convBlock3 = convolutional.Convolution2D(64,7,7,border_mode='same')(convBlock2)
convBlock3 = normalization.BatchNormalization(epsilon=1e-3)(convBlock3)
convBlock3 = core.Activation(lRelu)(convBlock3)
convBlock3 = core.Dropout(0.5)(convBlock3)
convBlock3 = pooling.MaxPooling2D(pool_size=(2,2),dim_ordering='tf')(convBlock3)


#end of ConvBlock3

#ApesBlock1 (64,5,0.5)

parallel2 = normalization.BatchNormalization(epsilon=1e-3)(convBlock3)
parallel1 = normalization.BatchNormalization(epsilon=1e-3)(convBlock3)
parallel1 = core.Activation(lRelu)(parallel1)
parallel1 = convolutional.Convolution2D(64,1,1,border_mode='same')(parallel1)
parallel1 = normalization.BatchNormalization(epsilon=1e-3)(parallel1)
parallel1 = core.Activation(lRelu)(parallel1)
parallel1 = convolutional.Convolution2D(64,7,7,border_mode='same')(parallel1)
parallel1 = normalization.BatchNormalization(epsilon=1e-3)(parallel1)

apesblock1 = merge([parallel1,parallel2],mode='sum')
apesblock1 = core.Activation(lRelu)(apesblock1)
apesblock1 = core.Dropout(0.5)(apesblock1)

#apesblock2 (64,5,0.5)

parallel2 = normalization.BatchNormalization(epsilon=1e-3)(apesblock1)
parallel1 = normalization.BatchNormalization(epsilon=1e-3)(apesblock1)
parallel1 = core.Activation(lRelu)(parallel1)
parallel1 = convolutional.Convolution2D(64,1,1,border_mode='same')(parallel1)
parallel1 = normalization.BatchNormalization(epsilon=1e-3)(parallel1)
parallel1 = core.Activation(lRelu)(parallel1)
parallel1 = convolutional.Convolution2D(64,7,7,border_mode='same')(parallel1)
parallel1 = normalization.BatchNormalization(epsilon=1e-3)(parallel1)

apesblock2 = merge([parallel1,parallel2],mode='sum')
apesblock2 = core.Activation(lRelu)(apesblock2)
apesblock2 = core.Dropout(0.5)(apesblock2)

'''END    OF   ENCODER    NETWORK'''


#Decoder Network with convBlock(64,7,0.5)

decoder1 = convolutional.UpSampling2D(size=(2,2))(apesblock2)
decoder1 = convolutional.Convolution2D(64,7,7,border_mode='same')(decoder1)
decoder1 = normalization.BatchNormalization(epsilon=1e-3)(decoder1)
decoder1 = core.Activation(lRelu)(decoder1)
decoder1 = core.Dropout(0.5)(decoder1)


#Decoder2 with conv block (16,7,0.5)

decoder2 = convolutional.UpSampling2D(size=(2,2))(decoder1)
decoder2 = convolutional.Convolution2D(16,7,7,border_mode='same')(decoder2)
decoder2 = normalization.BatchNormalization(epsilon=1e-3)(decoder2)
decoder2 = core.Activation(lRelu)(decoder2)
decoder2 = core.Dropout(0.5)(decoder2)

#Decoder3 with convBlock (8,7,0.5)

decoder3 = convolutional.UpSampling2D(size=(2,2))(decoder2)
decoder3 = convolutional.Convolution2D(8,7,7,border_mode='same')(decoder3)
decoder3 = normalization.BatchNormalization(epsilon=1e-3)(decoder3)
decoder3 = core.Activation(lRelu)(decoder3)
decoder3 = core.Dropout(0.5)(decoder3)

'''END   OF   DECODER   NETWORK'''

finalImage = convolutional.Convolution2D(12,1,1,border_mode='valid')(decoder3)

classification = Reshape((360 * 480, 12), input_shape=(360, 480, 12))(finalImage)
classification = Activation('softmax')(classification)

sgd = SGD(lr=0.1,momentum=0.9)


model = Model(input=inputImage,output=classification)
model.compile(optimizer=sgd,loss='categorical_crossentropy',metrics=['accuracy'])

plot(model, to_file='model.png')
'''
current_dir = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(current_dir, "apesnet.png")
plot(model, to_file=model_path, show_shapes=True)
'''
nb_epoch = 1000
batch_size = 1

model.load_weights('apesnet-003.hdf5')
history = model.fit(train_data, train_labels, batch_size=batch_size, nb_epoch=nb_epoch,
verbose=1, class_weight=class_weighting )#, validation_data=(X_test, X_test))

model.save_weights('apesnet-004.hdf5')




#model.load_weights('apesnet-001.hdf5')
import matplotlib.pyplot as plt
import cv2

Sky = [128, 128, 128]
Building = [128, 0, 0]
Pole = [192, 192, 128]
Road_marking = [255, 69, 0]
Road = [128, 64, 128]
Pavement = [60, 40, 222]
Tree = [128, 128, 0]
SignSymbol = [192, 128, 128]
Fence = [64, 64, 128]
Car = [64, 0, 128]
Pedestrian = [64, 64, 0]
Bicyclist = [0, 128, 192]
Unlabelled = [0, 0, 0]

label_colours = np.array([Sky, Building, Pole, Road, Pavement,
                          Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled])


def visualize(temp, plot=True):
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(0, 11):
        r[temp == l] = label_colours[l, 0]
        g[temp == l] = label_colours[l, 1]
        b[temp == l] = label_colours[l, 2]

    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    rgb[:, :, 0] = (r / 255.0)  # [:,:,0]
    rgb[:, :, 1] = (g / 255.0)  # [:,:,1]
    rgb[:, :, 2] = (b / 255.0)  # [:,:,2]
    if plot:
        plt.imshow(rgb)
    else:
        return rgb

output = model.predict(train_data)
print(output.shape)
pred = visualize(np.argmax(output,axis=2).reshape((360,480)), False)
plt.imshow(pred)
plt.show()
plt.figure(0)
#tr = visualize(np.argmax(train_labels,axis=1).reshape((360,480)), False)
#plt.imshow(tr)
#plt.show()'''