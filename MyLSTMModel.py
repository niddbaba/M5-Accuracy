import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM


def reshape_data(shape):
    i = 1
    while True:
        if i*56 > shape:
            return (i-1)*56
        else:
            i += 1


# 划分数据为训练集、测试集
def split_data(train_data):
    train = train_data.iloc[:-56]
    test = train_data.iloc[-56:]
    return train[train.columns[:-1]], train[train.columns[-1]], test[test.columns[:-1]]


def train_and_predict(train_data):
    train_X, train_y, test_X = split_data(train_data)
    print(train_X.shape, test_X.shape)
    model = Sequential()
    model.add(LSTM(50, activation='tanh', input_shape=(
        56, train_X.shape[1]), return_sequences=True))  # 设input_shape = (56, train_X.shape[1])表示每次考察56个时间步长，
    model.add(Dense(1))

    size = reshape_data(train_X.shape[0])

    train_X = train_X.iloc[-size:].values.reshape(
        (train_X.shape[0]//56, 56, train_X.shape[1]))
    train_y = train_y.iloc[-size:].values.reshape(
        (train_y.shape[0]//56, 56, 1))
    test_X = test_X.values.reshape((1, 56, test_X.shape[1]))  # 考察56个滞后时间步长

    model.compile(loss='mae', optimizer='adam')
    model.fit(train_X, train_y, epochs=50, batch_size=1, verbose=0)
    return model.predict(test_X, verbose=0)
    '''
    ValueError: Error when checking input: expected lstm_1_input to have shape (1012, 15) but got array with shape (56, 15)
    对于一个样本，时间步长设定为1012，需要转化
    '''
