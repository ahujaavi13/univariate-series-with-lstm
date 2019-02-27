# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 03:17:24 2019

@author: abhishek
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from matplotlib import pyplot

def preprocess_data(data_location,series_column,time_column):
    
    series = pd.read_csv(data_location, index_col = time_column)[[series_column]]
    
    # normalize features - 
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled = scaler.fit_transform(series)
    series = pd.DataFrame(scaled)
    
    return series,scaler

def reshape_by_window(preprocessed_data,window_size):
    
    series_s = preprocessed_data.copy()
    for i in range(window_size):
        preprocessed_data = pd.concat([preprocessed_data, \
                                       series_s.shift(-(i+1))], axis = 1)
    
    preprocessed_data.dropna(axis=0, inplace=True)
    
    return preprocessed_data

def univariate_processing(data_location,series_column,time_column,window_size):
    
    preprocessed_data,scaler = preprocess_data(data_location,series_column,time_column)
    print("Preprocessing Done")
    series = reshape_by_window(preprocessed_data,window_size)
    print("Reshaping Done")
    
    return series,scaler

def full_data_prep(series):

    train_size = series.shape[0]
    
    train = series.iloc[:train_size,:]
    
    train = train.values
    
    train = train.reshape(train.shape[0],train.shape[1],1)
    
    return train  

def train_data_prep(series,training_data_perc,shuffle_train):
    
    
    train_size = round((training_data_perc/100)*series.shape[0])
    
    train = series.iloc[:train_size, :]
    
    if shuffle_train == True: 
        train = shuffle(train)
    
    train_X = train.iloc[:,:-1]
    train_y = train.iloc[:,-1]
    
    train_X = train_X.values
    train_y = train_y.values
    
    train_X = train_X.reshape(train_X.shape[0],train_X.shape[1],1)
    
    return train_X,train_y

def test_data_prep(series,training_data_perc):

    train_size = round((training_data_perc/100)*series.shape[0])
    
    test = series.iloc[train_size:,:]
    
    test_X = test.iloc[:,:-1]
    test_y = test.iloc[:,-1]
    
    test_X = test_X.values
    test_y = test_y.values
    
    test_X = test_X.reshape(test_X.shape[0],test_X.shape[1],1)
    
    return test_X,test_y

def moving_test_window_preds(model,scalar,data_x,n_future_preds):

    ''' n_future_preds - Represents the number of future predictions we want to make
                         This coincides with the number of windows that we will move forward
                         on the test data
    '''
    preds_moving = []                                       # Use this to store the prediction made on each test window
    moving_test_window = [data_x[-1,1:].tolist()]             # Creating the first test window
    moving_test_window = np.array(moving_test_window)       # Making it an numpy array
    
    for i in range(n_future_preds):
        preds_one_step = model.predict(moving_test_window)  # Note that this is already a scaled prediction so no need to rescale this
        preds_moving.append(preds_one_step[0,0])            # get the value from the numpy 2D array and append to predictions
        preds_one_step = preds_one_step.reshape(1,1,1)      # Reshaping the prediction to 3D array for concatenation with moving test window
        moving_test_window = np.concatenate((moving_test_window[:,1:,:], preds_one_step), axis=1) # This is the new moving test window, where the first element from the window has been removed and the prediction  has been appended to the end
    
    preds_moving = pd.Series( (v for v in preds_moving) )
    preds_moving = scalar.inverse_transform(preds_moving.reshape(-1,1))
    
    return preds_moving

def prediction_plot(predicted_data,plot_title,save_loc):
    pyplot.plot(predicted_data)
    pyplot.legend(predicted_data.columns)
    pyplot.title(plot_title)
    pyplot.savefig(save_loc)
    
    return "Plot Saved"