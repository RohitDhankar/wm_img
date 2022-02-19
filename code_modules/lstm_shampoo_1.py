import pandas as pd 
from datetime import datetime 


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

