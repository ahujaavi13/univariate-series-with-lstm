# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 03:17:24 2019

@author: abhishek
"""

from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential


def kerasmodelbuilder(window_size):
    model = Sequential()
    model.add(LSTM(input_shape = (window_size,1),units = window_size, \
                                    return_sequences = True))
    model.add(LSTM(512))
    model.add(Dense(1))
    model.add(Activation("linear"))
    model.compile(loss="mse", optimizer="adam")
    model.summary()
 
    return model