## Preparing data and training it with an ARIMA model
#importing the prerequisite libraries
import pandas as pd
from sklearn import preprocessing
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
from matplotlib import pyplot

#reading the dataset
dataset = pd.read_csv('train.csv' )
features = pd.read_csv('features.csv')
stores = pd.read_csv('stores.csv')
dArima=pd.read_csv("dfArima.csv")
dArima['Group'].unique()
dArima0 = dArima[dArima.Group == 3331]

dArima1 =dArima0[['Date','Weekly_Sales']]
dArima2=dArima0[['Weekly_Sales']]
dArima1 = dArima1.reset_index(drop=True)

Norm_op=dArima1[['Weekly_Sales']]
x = Norm_op.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x = x.reshape(-1, 1)
x_scaled = min_max_scaler.fit_transform(x)
Norm_op_Nor = pd.DataFrame(x_scaled,columns=Norm_op.columns)
dArima1['Weekly_Sales'] = Norm_op_Nor

test1= dArima2[-12:]

# split into train and test sets
X = dArima2.values.reshape(-1,1)
train, test = X[0:-12], X[-12:]
history = [x for x in train]
predictions = list()

# walk-forward validation
for t in range(len(test)):
	# fit model
	model = ARIMA(history, order=(4,0,2))
	model_fit = model.fit()
	# one step forecast
	mfit = model_fit.forecast()[0]
	# store forecast and ob
	predictions.append(mfit)
	history.append(test[t])

# evaluate forecasts
mae = mean_absolute_error(test, predictions)    
mse = mean_squared_error(test, predictions)    
rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)
print('Test MSE: %.3f' % mse )
print('Test MAE: %.3f' % mae )
pyplot.figure(figsize=(16,9))
pyplot.plot(train)
pyplot.legend(['train'], loc='upper left')
pyplot.show()

# plot forecasts against actual outcomes
pyplot.figure(figsize=(16,9))
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.legend(['test', 'predictions'], loc='upper left')
pyplot.show()

#Arima Plot
forecast = pd.DataFrame(predictions,index = test1.index,columns=['Prediction'])

#printing actual and predicted values
dArimaFinal = pd.DataFrame({'Actual': test1.values.flatten(), 'Predicted': forecast.values.flatten()})
print(dArimaFinal)
