## Training the model with the final tuned neural network model base on the selected hyperparameters from grid search

#importing the prerequisite libraries
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
#from xgboost import XGBRegressor
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
#from keras.utils import to_categorical
#from keras.wrappers.scikit_learn import KerasRegressor
from keras import metrics
#from keras import backend
from matplotlib import pyplot

#importing the datasets
dataset = pd.read_csv('train.csv' )
features = pd.read_csv('features.csv')
stores = pd.read_csv('stores.csv')
dfArima=pd.read_csv("dfArima.csv")
#removing IsHoliday column from 'features' dataset
features=features.drop(columns=['IsHoliday'])

#merging the data set
df = dataset.merge(stores, how='left').merge(features, how='left')
#checking the shape of the dataset
df.shape
df.dtypes
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
df2.dtypes
dfA=df1[['Date','Store','Dept','Weekly_Sales']]
dfA.dtypes
df2['Date'] = le.fit_transform(df1['Date'])
df2['Date'].unique()

#listing the columns
cols = list(df2.columns.values)
cols
df2 = df2[['Date','Store','Dept','Type_A','Type_B','Type_C','IsHoliday','Size','Temperature','Fuel_Price','MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5','Unemployment','CPI','Weekly_Sales']]
df2['Store'].unique()
df2['Dept'].unique()
df2.dtypes

#check max and min
df2['Weekly_Sales'].max()
df2['Weekly_Sales'].min()
df2['Weekly_Sales'].mean
df2['Weekly_Sales']

#removing outlier section
dfo=df2[df2.Weekly_Sales<=100000]
dfo.shape
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

X2 = X1.iloc[:10000,:17]
Y2 = Y1.iloc[:10000,:]# normailized output

X_train, X_validation, Y_train, Y_validation = train_test_split(X2,Y2, test_size = 0.20, random_state = 10)

#Implementing the final Neural Network model

ip_testNor=X_validation
ip_trainNor=X_train
op_train1 =Y_train
op_test= Y_validation
op_test['Weekly_Sales'].mean()

    
model = Sequential()
model.add(Dense(17, input_dim =17, kernel_initializer='he_uniform', activation='relu'))
model.add(Dense(10, kernel_initializer='he_uniform', activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='mean_absolute_error' ,optimizer='Nadam' ,metrics=['mean_absolute_error','mean_squared_error'])#, 'mean_absolute_percentage_error', 'cosine_proximity'])
history = model.fit(ip_trainNor,op_train1,epochs= 100, batch_size= 10)


_,train_mae,train_mse = model.evaluate(ip_trainNor,op_train1)
print('mean_absolute_error: %.2f' % (train_mae)) 
print('mean_square_error: %.2f' % (train_mse)) 
  
  
_,test_mae,test_mse = model.evaluate(ip_testNor,op_test)
print('mean_absolute_error: %.2f' % (test_mae))
print('mean_square_error: %.2f' % (test_mse))

print('Train: %.3f, Test: %.3f' % (train_mae, test_mae))
print('Train: %.3f, Test: %.3f' % (train_mse, test_mse))
predictions = model.predict(ip_testNor)

#predictions = model.predict_classes(ip_testNor)
predictions.shape
op_test.shape
op_test =np.asarray(op_test, dtype="float")
#op_test = np.argmax(op_test, axis=-1)

#op_test = to_categorical(Y_validation)
op_test.shape
ip_testNor.shape

print(history.history.keys())

# summarize history for accuracy
plt.plot(history.history['mean_absolute_error'])
#plt.plot(history.history['mean_absolute_error'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend([ 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend([ 'test'], loc='upper left')
plt.show()



for i in range(20):
    print('predicted = %.5f ,(expected = %.5f)' %(predictions[i], op_test[i]))

predictions =predictions.reshape(-1,1)  
op_test=op_test.reshape(-1,1)

# plot the metrics
pyplot.plot(history.history['mean_squared_error'])
pyplot.plot(history.history['mean_absolute_error'])

pyplot.show()
print('Mean Absolute Error:', metrics.mean_absolute_error(predictions, op_test))  
print('Mean Squared Error:', metrics.mean_squared_error(predictions, op_test))  

