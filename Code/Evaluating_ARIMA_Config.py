## Evaluating and configuring ARIMA parameters (p,d,q)
#importing pre-requisite library
import pandas as pd
from sklearn import preprocessing
from statsmodels.tsa.arima_model import ARIMA
import warnings
from sklearn.metrics import mean_absolute_error

#reading the dataset
dataset = pd.read_csv('train.csv' )
features = pd.read_csv('features.csv')
stores = pd.read_csv('stores.csv')
dfA=pd.read_csv("dfArima.csv")

dfxxx = dfA[dfA.Group == 3331]
dfx1 =dfxxx[['Date','Weekly_Sales']]
dfx2=dfxxx[['Weekly_Sales']]
dfx1 = dfx1.reset_index(drop=True)

ip_train11=dfx1[['Weekly_Sales']]
x = ip_train11.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x = x.reshape(-1, 1)
x_scaled = min_max_scaler.fit_transform(x)
ip_trainNor = pd.DataFrame(x_scaled,columns=ip_train11.columns)
dfx1['Weekly_Sales'] = ip_trainNor

# evaluate an ARIMA model for a given order (p,d,q)
def evaluate_arima_model(X, arima_order):
	# prepare training dataset
	train_size = int(len(X) * 0.66)
	train, test = X[0:train_size], X[train_size:]
	history = [x for x in train]
	# make predictions
	predictions1 = list()
	for t in range(len(test)):
		model = ARIMA(history, order=arima_order)
		model_fit = model.fit(disp=0)
		yhat = model_fit.forecast()[0]
		predictions1.append(yhat)
		history.append(test[t])
	# calculate out of sample error
	error = mean_absolute_error(test, predictions1)
	return error
 
# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models1(dataset, p_values, d_values, q_values):
	dataset = dataset.astype('float32')
	best_score, best_cfg = float("inf"), None
	for p in p_values:
		for d in d_values:
			for q in q_values:
				order = (p,d,q)
				try:
					mae = evaluate_arima_model(dataset, order)
					if mae < best_score:
						best_score, best_cfg = mae, order
					print('ARIMA%s MAE=%.3f' % (order,mae))
				except:
					continue
	print('Best ARIMA%s MAE=%.3f' % (best_cfg, best_score))

dfx1 = dfx1.astype('float32')    
# evaluate parameters
p_values = [0, 1, 2, 4, 6, 8, 10]
d_values = range(0, 3)
q_values = range(0, 3)
warnings.filterwarnings("ignore")
evaluate_models1(dfx1.values.reshape(-1,1), p_values, d_values, q_values)