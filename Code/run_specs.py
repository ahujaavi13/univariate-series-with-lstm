# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 03:17:24 2019

@author: abhishek
"""

import subprocess

#The following code automatically detects root directory
wd  = subprocess.check_output('git rev-parse --show-toplevel', \
                               shell=True).decode('utf-8').strip()

#If you want to delete all previously stored logs and models
clear_log = "yes"
clear_model = "yes"

#######################Model Specifications######################

window_size = 64
n_future_preds = 200
series_column = "Passenger_Count"
time_column   = "Time"
MODEL_NAME = 'univariate-series-with-lstm'

######################Create Data################################

datasource = "/home/ubuntu/Data_Warehouse/univariate-time-series"
TRAIN_FILE = datasource + '/international-airline-passengers.csv'
training_data_perc = 80
validation_perc = 10

#####################Training Specifications#######################

epochs = 500
batch_size= 512

#################Model Saving####################################

saved_model_name = "univariate-series-predictor"

#################Predictions generation##########################

predictions_file_name = "predictions"
predictions_plot_name = "ActualvsPredicted"