# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 03:17:24 2019

@author: abhishek
"""

########################Import Reqisite Functions##########################

from run_specs import *
from helper_lstm import *
from model_builder import *
import pandas as pd
from sklearn.metrics import mean_squared_error


actual_data = pd.read_csv(TRAIN_FILE, index_col = time_column)

######################Create Data#########################################
processed_univariate,scalar = univariate_processing(data_location = TRAIN_FILE, \
                                                    series_column = series_column, \
                                                    time_column = time_column, \
                                                    window_size = window_size)

x_test,y_test = test_data_prep(series = processed_univariate, \
                                 training_data_perc = training_data_perc)

x_full = full_data_prep(series = processed_univariate)

######################Model Loading#########################################

from keras.models import load_model
model = load_model(wd + "/Model/" + saved_model_name +'.h5')

#####################Generate Predictions##################################

save_location = wd + "/Predictions/" 

#Validation
preds_val = model.predict(x_test)
preds_val = scalar.inverse_transform(preds_val)
actuals_val =  scalar.inverse_transform(y_test.reshape(-1,1))
error = mean_squared_error(actuals_val,preds_val)
print("Means Squared error on Holdout is :" + str(error) )

preds_val = [item for sublist in preds_val for item in sublist]
predicted_data_val = pd.concat([actual_data, \
                         pd.DataFrame({"Holdout_Prediction":preds_val},index = range(actual_data.shape[0] - len(preds_val),actual_data.shape[0]))], axis=1)


#Predict Ahead
preds_moving = moving_test_window_preds(model = model, \
                                        scalar = scalar, \
                                        data_x = x_full, \
                                        n_future_preds = n_future_preds)
preds_moving_list = [item for sublist in preds_moving for item in sublist]
predicted_data_forecast = pd.concat([predicted_data_val, \
            pd.DataFrame({"Prediction_ahead":preds_moving_list}, \
            index = range(predicted_data_val.shape[0]+1, \
                          predicted_data_val.shape[0]+1 + \
                          n_future_preds))] , axis=1)

predicted_data_forecast.to_csv(save_location + "Predictions.csv")

#Plotting Actual Vs predicted
prediction_plot(predicted_data = predicted_data_forecast, \
                plot_title = "Predicting for " + str(n_future_preds) + " time steps", \
                save_loc = save_location + predictions_plot_name + ".png"
                )

pyplot.plot(predicted_data_forecast)
pyplot.legend(predicted_data_forecast.columns)
pyplot.show()