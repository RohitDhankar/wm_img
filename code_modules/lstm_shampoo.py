import pandas as pd 
from datetime import datetime 
#from pandas import datetime
from sklearn.metrics import mean_squared_error
from math import sqrt
from matplotlib import pyplot


# load dataset
def parser(x):
	return datetime.strptime('190'+x, '%Y-%m')
series = pd.read_csv('../inPut_dir/shampoo.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
# summarize first few rows
print(series.head())

# split data into train and test
X = series.values
train, test = X[0:-12], X[-12:]
print("---shape---train",train.shape)
print("---shape---test",test.shape)

# walk-forward validation
history = [x for x in train]
predictions = list()

for i in range(len(test)):
	# make prediction
	predictions.append(history[-1])
	# observation
	history.append(test[i])
# report performance
rmse = sqrt(mean_squared_error(test, predictions))
print('RMSE: %.3f' % rmse)
# line plot of observed vs predicted
pyplot.plot(test)
pyplot.plot(predictions)
pyplot.show()

"""
LSTM Data Preparation - 
Before we can fit an LSTM model to the dataset, we must transform the data.

This section is broken down into three steps:

    Transform the time series into a supervised learning problem
    Transform the time series data so that it is stationary.
    Transform the observations to have a specific scale.

"""
#
"""
Transform Time Series to Supervised Learning
The LSTM model in Keras assumes that your data is divided into input (X) and output (y) components.
For a time series problem, we can achieve this by using the observation from the last time step (t-1) 
as the input and the observation at the current time step (t) as the output.
We can achieve this using the shift() function in Pandas that will push all values in a series down by 
a specified number places. 
We require a shift of 1 place, which will become the input variables.
The time series as it stands will be the output variables.
We can then concatenate these two series together to create a DataFrame ready 
for supervised learning. 
The pushed-down series will have a new position at the top with no value. 
A NaN (not a number) value will be used in this position. 
We will replace these NaN values with 0 values, which the LSTM model will have to 
learn as “the start of the series” or “I have no data here,” as a month with zero sales 
on this dataset has not been observed.
"""
#

# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
	df = pd.DataFrame(data)
	print("----df.shape_1---",df.shape)
	columns = [df.shift(i) for i in range(1, lag+1)]
	print("----type(columns--",type(columns))
	print("----columns--",columns)

	columns.append(df)
	print("----df.shape_2---",df.shape)
	df = pd.concat(columns, axis=1)
	df.fillna(0, inplace=True)
	return df

print(series.head())
concat_df = timeseries_to_supervised(series, lag=1)	
file_path = '../inPut_dir/'
concat_df.to_csv(file_path+"concat_df.csv")

# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return Series(diff)

# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]

### LSTM -- KERAS -- https://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/
# 		

