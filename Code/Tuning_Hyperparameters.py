## Performing grid search on several hyperparameter to tune the neural network model

#importing the prerequisite libraries
import pandas as pd
from sklearn import preprocessing
from sklearn import metrics
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
#from keras.utils import to_categorical
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor

#Define a model for batch size and epochs tuning
def create_model():
    model = Sequential()
    model.add(Dense(17, input_dim =17, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='mean_absolute_error' ,optimizer='adam' ,metrics=['mean_absolute_error','mean_squared_error', 'mean_absolute_percentage_error', 'cosine_proximity'])
    return model

#Define a model for optimizer tuning
def create_model1(optimizer='adam'):
    model = Sequential()
    model.add(Dense(17, input_dim =17, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='mean_absolute_error' ,optimizer= optimizer ,metrics=['mean_absolute_error','mean_squared_error', 'mean_absolute_percentage_error', 'cosine_proximity',])
    return model

#Define a model for initial weights tuning
def create_model2(init_mode='uniform'):
    model = Sequential()
    model.add(Dense(17, input_dim =17, kernel_initializer=init_mode, activation='relu'))
    model.add(Dense(10, kernel_initializer=init_mode, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='mean_absolute_error' ,optimizer= 'Nadam' ,metrics=['mean_absolute_error','mean_squared_error', 'mean_absolute_percentage_error', 'cosine_proximity'])
    return model

#Define a model for activation function tuning
def create_model3(activation='relu'):
    model = Sequential()
    model.add(Dense(17, input_dim =17, kernel_initializer='he_uniform', activation= activation))
    model.add(Dense(10, kernel_initializer='he_uniform', activation= activation))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='mean_absolute_error' ,optimizer= 'Nadam' ,metrics=['mean_absolute_error','mean_squared_error', 'mean_absolute_percentage_error', 'cosine_proximity'])
    return model

#Define a model neuron count tuning
def create_model4(neurons1=1, neurons2=1):
    model = Sequential()
    model.add(Dense(neurons1, input_dim =17, kernel_initializer='he_uniform', activation= 'relu'))
    model.add(Dense(neurons2, kernel_initializer='he_uniform', activation= 'relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='mean_absolute_error' ,optimizer= 'Nadam' ,metrics=['mean_absolute_error','mean_squared_error', 'mean_absolute_percentage_error', 'cosine_proximity'])
    return model

#reading the dataset
dataset = pd.read_csv("train.csv" )
features = pd.read_csv("features.csv" )
stores = pd.read_csv("stores.csv" )
#removing IsHoliday column from 'features' dataset
features=features.drop(columns=['IsHoliday'])
#merging the data set
df = dataset.merge(stores, how='left').merge(features, how='left')
#checking the shape of the dataset
df.shape
#checking if there is null values
df.isnull().sum()
#replacing null values to 0
df1 = df.fillna(0)



 #replacing true and false in IsHoliday column to 1 and 0  
df1['IsHoliday']= df1['IsHoliday'].replace(True,1)
df1['IsHoliday']= df1['IsHoliday'].replace(False,0)
#converting float to int 
df1['IsHoliday'] = df1['IsHoliday'].astype(int)
df1.dtypes
#unique values of a column
df1['Type'].unique()

le = preprocessing.LabelEncoder()
#convert the categorical columns into numeric
df1['Type'] = le.fit_transform(df1['Type'])
df1[["Type"]] = df1[["Type"]].apply(pd.to_numeric)
df2 = pd.get_dummies(df1, columns=["Type"], prefix=["Type"])
df2.rename(columns={'Type_0': 'Type_A', 'Type_1': 'Type_B', 'Type_2': 'Type_C'}, inplace=True)

#converting the target values to integers from flaot
df2['Weekly_Sales'] = df2['Weekly_Sales'].round().astype(int)
df2['Type_A'] = df2['Type_A'].round().astype(int) #to convert uint8 to int
df2['Type_B'] = df2['Type_B'].round().astype(int)
df2['Type_C'] = df2['Type_C'].round().astype(int)
df2['Date'] = le.fit_transform(df1['Date'])
df2 = df2[['Date','Store','Dept','Type_A','Type_B','Type_C','IsHoliday','Size','Temperature','Fuel_Price','MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5','Unemployment','CPI','Weekly_Sales']]

#removing outlier section
dfo=df2[df2.Weekly_Sales<=100000]

#Normalizing dataset input
ip_train11=dfo[['Date','Store','Dept','Type_A','Type_B','Type_C','IsHoliday','Size','Temperature','Fuel_Price','MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5','Unemployment','CPI']]
Xn= dfo.iloc[:,:17] #0:17 is the input col
x = ip_train11.values #returns a numpy array of wk values
min_max_scaler = preprocessing.MinMaxScaler()
#x = x.reshape(-1, 1) #to get x in a single column. -1, 1 to transpose :}
x_scaled = min_max_scaler.fit_transform(x)
ip_trainNor = pd.DataFrame(x_scaled,columns=ip_train11.columns) #turn array to dataframe
ip_trainNor.shape 
ip_trainNor.columns
ip_trainNor.dtypes
X1= ip_trainNor
#Normalizing dataset output
op_train11= dfo[['Weekly_Sales']]
Yn= df2.iloc[:,17:18] #17:18 is the weekly sales col
x = op_train11.values #returns a numpy array of wk values
min_max_scaler = preprocessing.MinMaxScaler()
x = x.reshape(-1, 1) #to get x in a single column. -1, 1 to transpose :}
x_scaled = min_max_scaler.fit_transform(x)
op_trainNor = pd.DataFrame(x_scaled,columns=op_train11.columns) #turn array to dataframe
op_trainNor.shape 
op_trainNor.columns
op_trainNor.dtypes
Y1= op_trainNor
ip_trainNor=ip_trainNor.iloc[:10000,:]
op_trainNor=op_trainNor.iloc[:10000,:]


#tuning batch size and epochs
model = KerasRegressor(build_fn=create_model, verbose=150)
# define the grid search parameters
batch_size = [10,  40, 100 ]
epochs = [50, 100,200]
param_grid = dict(batch_size=batch_size, epochs=epochs)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv=3, verbose=150)
grid_result = grid.fit(ip_trainNor, op_trainNor)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


#tuning optimizer
model = KerasRegressor(build_fn=create_model1, epochs=100, batch_size=20, verbose=0)
# define the grid search parameters
optimizer = [ 'SGD', 'RMSprop', 'Adagrad', 'Adadelta','Adam', 'Adamax', 'Nadam']
param_grid = dict(optimizer=optimizer)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv=3, verbose=150)
grid_result = grid.fit(ip_trainNor, op_trainNor)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


#tuning network weights initialization  
model = KerasRegressor(build_fn=create_model2, epochs=100, batch_size=20, verbose=0)
# define the grid search parameters
init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
param_grid = dict(init_mode=init_mode)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv=3, verbose=150)
grid_result = grid.fit(ip_trainNor, op_trainNor)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


#tuning activation
model = KerasRegressor(build_fn=create_model3, epochs=150, batch_size=20, verbose=0)
# define the grid search parameters
activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
param_grid = dict(activation=activation)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv=3, verbose=150)
grid_result = grid.fit(ip_trainNor, op_trainNor)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


#tuning number of neurons
model = KerasRegressor(build_fn=create_model4, epochs=150, batch_size=20, verbose=0)
# define the grid search parameters
neurons1 = [ 20,  30, 50]
neurons2 = [ 20,  30, 50]
param_grid = dict(neurons1=neurons1,neurons2=neurons2 )
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv=3, verbose=150)
grid_result = grid.fit(ip_trainNor, op_trainNor)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
