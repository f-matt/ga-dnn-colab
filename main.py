#! /usr/bin/python
# -*- coding:utf-8 -*-

from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.callbacks import EarlyStopping
import numpy as np
import matplotlib.pyplot as plt
import cv2

from copy import deepcopy

from samba.dcerpc.xattr import EA

TRAIN_FILE = "data/ar-tracking/train.csv"
TEST_FILE = "data/ar-tracking/test.csv"

TRAIN_PATTERNS = 1024
TEST_PATTERNS = 256

BATCH_SIZE = 64
EPOCHS = 100

IMG_WIDTH = 100
IMG_HEIGHT = 100
IMG_CHANNELS = 3

NORM_FACTOR = 50



#
# Train net
#
def train_net():
    train_file = open(TRAIN_FILE, 'r')
    train_lines = train_file.readlines()
    train_file.close()

    test_file = open(TEST_FILE, 'r')
    test_lines = test_file.readlines()
    test_file.close()

    x_train = list()
    y_train = list()

    x_test = list()
    y_test = list()

    for i in range(TRAIN_PATTERNS):
        fields = train_lines[i].split(';')
        img = cv2.imread(fields[0])

        x_train.append(deepcopy(img))
        y_train.append(map(float, fields[1:]))

    for i in range(TEST_PATTERNS):
        fields = test_lines[i].split(';')
        img = cv2.imread(fields[0])

        x_test.append(deepcopy(img))
        y_test.append(map(float, fields[1:]))

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    x_test = np.array(x_test)
    y_test = np.array(y_test)

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH)
        x_test = x_test.reshape(x_test.shape[0], IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH)
        input_shape = (IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH)
    else:
        x_train = x_train.reshape(x_train.shape[0], IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
        x_test = x_test.reshape(x_test.shape[0], IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
        input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

    x_train = x_train.astype('float32')
    x_train /= 255

    mean_image = np.sum(x_train, 0) / x_train.shape[0]

    x_train -= mean_image

    x_test = x_test.astype('float32')
    x_test /= 255

    x_test -= mean_image

    y_train = y_train.astype('float32')
    y_train = (y_train - NORM_FACTOR) / NORM_FACTOR

    y_test = y_test.astype('float32')
    y_test = (y_test - NORM_FACTOR) / NORM_FACTOR

    # Model: CR;16;3;1;2
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(3, 3),
                     activation='relu',
                     strides=1,
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(8, kernel_initializer="uniform"))

    # Compila o modelo
    model.compile(loss='mean_squared_error', optimizer='adam')

    # Treina o modelo
    history = model.fit(x_train,
                        y_train,
                        epochs=EPOCHS,
                        batch_size=BATCH_SIZE,
                        validation_data=(x_test, y_test),
                        callbacks=[EarlyStopping(monitor='val_loss', patience=10, verbose=0)])

    # Exporta o modelo
    # model.save('model.h5')

    print(history.history.keys())
    #    print(history.history['acc'])

    #   plt.plot(history.history['acc'])
    plt.plot(history.history['loss'])

    plt.show()

    test_loss = 0

    for i in range(TEST_PATTERNS):
        pred = model.predict(x=x_test[i], batch_size=1, verbose=0)

        loss = 0.5 * np.sum(np.sqr(pred - y_test[i]))

        test_loss += loss

    test_loss /= TEST_PATTERNS

    print "Test loss: ", test_loss

    # Avalia o modelo


#  scores = model.evaluate(X, Y)
#  print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


#
# Teste da rede com padrões desconhecidos
#
def test_net():
    fid = open('test.csv', 'r')
    lines = fid.readlines()
    fid.close()

    dataset = []

    for line in lines:
        dataset.append(list(map(float, line.rstrip('\n').split(','))))

    dataset = np.array(dataset)

    n_test_patterns = 100

    # Divide o dataset em entradas (X) e saídas (Y)
    X = dataset[0:n_test_patterns, 0:41]
    Y = dataset[0:n_test_patterns, 41]

    model = load_model('model.h5')

    pred = model.predict(x=X, batch_size=1, verbose=0)

    n_correct = 0
    n_wrong = 0

    for i in range(len(pred)):
        y_pred = int(round(pred[i][0]))

        if y_pred == Y[i]:
            n_correct += 1
        else:
            n_wrong += 1

    acc = float(n_correct) / (n_correct + n_wrong)

    print "Acc: ", acc * 100, "%"


if __name__ == "__main__":
    # Fixa o gerador de números aleatórios
    np.random.seed(2)

    # split_datasets()
    train_net()
    # test_net()

