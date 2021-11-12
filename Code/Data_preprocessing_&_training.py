## Performing data preprocessing, cleaning and training the models with respect to set of Machine learning Algorithms
#importing prerequisite libraries
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import ExtraTreeRegressor
from sklearn.base import clone
from sklearn import metrics
from keras.utils import to_categorical
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

#reading the dataset
dataset = pd.read_csv('train.csv' )
features = pd.read_csv('features.csv')
stores = pd.read_csv('stores.csv')
features=features.drop(columns=['IsHoliday'])
#merging the data set
data_set = dataset.merge(stores, how='left').merge(features, how='left')
#checking if there is null values
data_set.isnull().sum()
#replacing null values to 0
data_set1 = data_set.fillna(0)

#replacing true and false in IsHoliday column to 1 and 0  
data_set1['IsHoliday']= data_set1['IsHoliday'].replace(True,1)
data_set1['IsHoliday']= data_set1['IsHoliday'].replace(False,0)
#converting float to int 
data_set1['IsHoliday'] = data_set1['IsHoliday'].astype(int)
Enco = preprocessing.LabelEncoder()

#convert the categorical columns into numeric
data_set1['Type'] = Enco.fit_transform(data_set1['Type'])
data_set1[["Type"]] = data_set1[["Type"]].apply(pd.to_numeric)
data_set2 = pd.get_dummies(data_set1, columns=["Type"], prefix=["Type"])
data_set2.rename(columns={'Type_0': 'Type_A', 'Type_1': 'Type_B', 'Type_2': 'Type_C'}, inplace=True)

#converting the target values to integers from flaot
data_set2['Weekly_Sales'] = data_set2['Weekly_Sales'].round().astype(int)
data_set2['Type_A'] = data_set2['Type_A'].round().astype(int) #to convert uint8 to int
data_set2['Type_B'] = data_set2['Type_B'].round().astype(int)
data_set2['Type_C'] = data_set2['Type_C'].round().astype(int)
data_set2['Date'] = Enco.fit_transform(data_set1['Date'])

#listing the columns
cols = list(data_set2.columns.values)
data_set2 = data_set2[['Date','Store','Dept','Type_A','Type_B','Type_C','IsHoliday','Size','Temperature','Fuel_Price','MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5','Unemployment','CPI','Weekly_Sales']]

data_seto=data_set2[data_set2.Weekly_Sales<=100000]

#Normalizing the dataset output
Norm_op= data_seto[['Weekly_Sales']]
Yn= data_set2.iloc[:,17:18] #17:18 is the weekly sales col
x = Norm_op.values #returns a numpy array of wk values
min_max_scaler = preprocessing.MinMaxScaler()
x = x.reshape(-1, 1) #to get x in a single column. -1, 1 to transpose :}
x_scaled = min_max_scaler.fit_transform(x)
Norm_op_Nor = pd.DataFrame(x_scaled,columns=Norm_op.columns) #turn array to dataframe
Y1= Norm_op_Nor

#Feature or attribute reduction process incorporating 2 methods RFE and univariate selection
# 1. Recursive Feature Elimination
#preparing data for RFE
X = data_set2.iloc[:10000,:17]
Y = data_seto.iloc[:9700,17:18]

df3=data_seto.loc[:10000,:]

#Setting the data into an array form for feature extraction
array = df3.values
X_t = array[:,0:17]#we gonna take from 0 to 16
Y_t = array[:,17] #taking output i.e. col 17
Y_t = np.asarray(df3['Weekly_Sales'], dtype="int") #to get into int

#performing feature extraction
model = RandomForestRegressor(n_estimators=50)
rfe = RFE(model, 13, verbose=150) #k=5,so 5 features; verbose shows sth on console
X_rfe = rfe.fit_transform(X_t,Y_t)
fit = rfe.fit(X_t, Y_t)
print("Best features chosen by RFE: \n")
for i in X.columns[rfe.support_]:
    print(i)

#number of features to be short-listed
nof_list=np.arange(1,17)            
high_score=0

#Variable to store the optimum features
nof=0           
score_list =[]
for n in range(len(nof_list)):
    X_train, X_test, y_train, y_test = train_test_split(X_t,Y_t, test_size = 0.2, random_state = 0)
    model = RandomForestRegressor(n_estimators=50)
    rfe = RFE(model,nof_list[n],verbose=150)
    X_train_rfe = rfe.fit_transform(X_train,y_train)
    X_test_rfe = rfe.transform(X_test)
    model.fit(X_train_rfe,y_train)
    score = model.score(X_test_rfe,y_test)
    score_list.append(score)
    if(score>high_score):
        high_score = score
        nof = nof_list[n]
print("Optimum number of features: %d" %nof)
print("Score with %d features: %f" % (nof, high_score))

# 2. Univariate Selection
test = SelectKBest(score_func=chi2, k=13)
fit = test.fit(X_t, Y_t)
# summarize scores
np.set_printoptions(precision=3)
print(fit.scores_)
features = fit.transform(X)
# summarize selected features
print(features[0:5,:])
diccionario = {key:value for (key, value) in zip(test.scores_, X.columns)}
sorted(diccionario.items())"""

Xrfe=X[['Date','Dept','IsHoliday','Size','Temperature','Fuel_Price','MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5','Unemployment','CPI']]
X_rfe1=pd.DataFrame(X_rfe,columns=Xrfe.columns)

value = {key:value for (key, value) in zip(rfe.ranking_, X_rfe1.columns)}
sorted(value.items())
print("Num Features: %d" % fit.n_features_)
print("Selected Features: %s" % fit.support_)
print("Feature Ranking: %s" % fit.ranking_)

Y2 = Y1.iloc[:9700,:]# normailized output

X_train, X_validation, Y_train, Y_validation = train_test_split(X_rfe1,Y2, test_size = 0.20, random_state = 10)


#Function to prepare a list of machine learning models
def get_models(models=dict()):
    models['Linear_Regression'] = LinearRegression()
    models['Decision_Tree_Regressor'] = DecisionTreeRegressor()
    models['K_Neighbors_Regressor'] = KNeighborsRegressor(n_neighbors=7)
    models['Extra_Tree_Regressor'] = ExtraTreeRegressor()
    return models

#Function to evaluate a suite of models
def evaluate_models(models, X_train, Y_train, X_validation, Y_validation):
	for name, model in models.items():
		# fit models
		fits = fit_model(model, X_train, Y_train)
		# make predictions
		predictions = make_predictions(name, fits, X_validation)
		# evaluate forecast
		#Errors = evaluate(name,  Y_validation, predictions)
		# summarize forecast
		#summarize_error(name, Errors)

#Function train the set of models
def fit_model(model, X_train, Y_train):
	# clone the model configuration
	local_model = clone(model)
	# fit the model
	local_model.fit(X_train, Y_train)
	return local_model

#Function to perform the predictions
def make_predictions(name, model, X_validation):    
        predictions = model.predict(X_validation)
        predictions = predictions.reshape(1940,1)#2000
        results = get_results(name, Y_validation, predictions)    
        return results
    

        
def get_results(name, Y_validation, predictions):
    #Accuracy = accuracy_score(Y_validation, predictions)
    #Y_validation1=np.asarray(Y_validation, dtype="float")
    #predictions=pd.DataFrame(predictions, columns=Y_validation.columns)
    RMSE     = np.sqrt(metrics.mean_squared_error(Y_validation, predictions))
    MAE      = metrics.mean_absolute_error(Y_validation, predictions) 
    MSE      = metrics.mean_squared_error(Y_validation, predictions)  
    print('Model: %s' % (name))
    #print('Accuracy: %.3f' % (Accuracy))
    print('Mean Absolute Error:  %.3f' % (MAE))   
    print('Mean Squared Error:  %.3f'  % (MSE))
    print('Root Mean Squared Error:  %.3f' % (RMSE))
    dfg=pd.DataFrame(predictions)
    dff1 = pd.DataFrame({'Actual': Y_validation.values.flatten(), 'Predicted': dfg.values.flatten()})
    dp1 = dff1.head(20)
    dp1.plot(kind='bar', title=name, figsize=(16,10))
     
    print(dff1.head(10))

# prepare list of models
models = get_models()
# evaluate models
evaluate_models(models, X_train, Y_train, X_validation, Y_validation)    


