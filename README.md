# **Univariate Series Forecasting with LSTM**

In this repository I have tried to build a forecasting algorithm that can predict a series for example passenger count
in airlines across years

## **Folder Guide**
```
|-----univariate-series-with-lstm  
		|----Code                                     #Codes for running Model			                      
			|----helper_lstm.py
			|----model_builder.py
			|----run_specs.py
			|----train_model.py
			|----test_model.py
			|----train_log.out
			|----test_log.out
		|----Log                                      #For Logging Tensorboard Output
		|----Model                                    #For saving model  
		|----Predictions                              #For Saving Predictions
		|----Sample_Data                              #Sample Data used for Modeling 
```
## Data

In this case I have tried to predict count of passengers in airline over a certain time period.  

Once the algorithm is ready and tuned properly it will do forecasting as it has been illustrated below from a model run

<img src="https://raw.githubusercontent.com/ahujaavi13/univariate-series-with-lstm/master/Predictions/ActualvsPredicted.png" width=350 height = 300>

#### Note:
In this case the model results might vary to some degree with rerun due to random initialization.
The current result being displayed is based on the model saved in folder `Model`. Will try to work on a model setting that provides more stability as an improvement for this repository
