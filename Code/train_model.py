# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 03:17:24 2019

@author: abhishek
"""

########################Import Reqisite Packages##########################

from run_specs import * 
from helper_lstm import *
from model_builder import *
from keras.callbacks import TensorBoard as tb
import os


#######################Remove old Logs and Models#######################

if clear_log == "yes":
	folder = wd +  "/Log/" + MODEL_NAME
	for the_file in os.listdir(folder):
		file_path = os.path.join(folder, the_file)
		try:
			if os.path.isfile(file_path):
				os.unlink(file_path)
		except Exception as e:
			print(str(e) + " does not exist")
			
if clear_model == "yes":
	folder = wd +  "/Model/" 
	for the_file in os.listdir(folder):
		file_path = os.path.join(folder, the_file)
		try:
			if os.path.isfile(file_path):
				os.unlink(file_path)
		except Exception as e:
			print(str(e) + " does not exist")

######################Create Data#########################################
processed_univariate,scaler = univariate_processing(data_location = TRAIN_FILE, \
                                                    series_column = series_column, \
                                                    time_column = time_column, \
                                                    window_size = window_size)

x_train,y_train= train_data_prep(series = processed_univariate, \
                                 training_data_perc = training_data_perc, \
                                 shuffle_train = True
                                 )

######################Model Fitting#########################################

tbCallBack = tb(log_dir= wd +  "/Log/" + MODEL_NAME,histogram_freq=0, batch_size=32, write_graph=True, write_grads= True, write_images=True)

model = kerasmodelbuilder(window_size)
model.fit(x_train, \
          y_train, \
          batch_size=batch_size, \
          epochs=epochs, \
          validation_split=0.1, \
          callbacks = [tbCallBack]
          )

from keras.models import load_model
model.save(wd + "/Model/" + saved_model_name + ".h5")

print("Model Training Complete")