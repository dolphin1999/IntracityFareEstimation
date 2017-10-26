


#imports
from sklearn.linear_model import (
    Ridge, LinearRegression,Lasso, TheilSenRegressor, RANSACRegressor, HuberRegressor, LassoCV)
from sklearn.model_selection import cross_val_score
from scipy.stats import norm
#from sklearn.cross_validation import cross_val_score
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
import pandas as pd
from pandas import Series, DataFrame
from sklearn import ensemble

#numpy, matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
#get data

fareTrain = pd.read_csv("intracity_fare_train.csv")
fareTest = pd.read_csv("intracity_fare_test.csv")

#print fareTrain.head()
#print fareTrain.info()
id1 = (fareTest["ID"])[0:20000]
id2 = (fareTrain["ID"])[0:7000]




#print fareTrain.groupby('VEHICLE_TYPE').count()
#print fareTest.groupby('VEHICLE_TYPE').count()

#print fareTrain.describe(include='all')

#print list(set(fareTrain.VEHICLE_TYPE))

fullData = [fareTrain,fareTest]


#print encoded_Vehicle2[:20000]

vehicle_mapping = {"Taxi non ac" : 12, 
                  "AC bus" : 17
                  , "taxi ac":20, 
                  "mini bus":8, 
                  "AUTO RICKSHAW":200, 
                  "taxi Non ac":12, 
                  "taxi non ac":12, 
                  "Bus":5, 
                  "Taxi AC":20, 
                  "Ac Bus":17, 
                  "Mini bus":8, 
                  "Taxi non Ac":12, 
                  "Mini Bus":8, 
                  "Auto rickshaw":200, 
                  "taxi non Ac":12, 
                  "Metro":4, 
                  "bus":5, 
                  "AC Bus":17, 
                  "auto rickshaw":200, 
                  "Taxi Non AC":12, 
                  "Auto Rickshaw":200, 
                  "taxi Non Ac":12, 
                  "metro":4, 
                  "ac bus":17, 
                  "mini Bus":8}



#fareTrain['VEHICLE_TYPE'] = encoded_Vehicle1
#fareTest['VEHICLE_TYPE'] = encoded_Vehicle2

for data in fullData:
    data["VEHICLE_TYPE"] = data["VEHICLE_TYPE"].map(vehicle_mapping)
#print fareTrain.head()
#print fareTest.head()   
    
'''    
fareTrain["TRAFFIC_STUCK_TIME"] = fareTrain["TRAFFIC_STUCK_TIME"]/2
fareTrain["DISTANCE"] = fareTrain["DISTANCE"]/2
fareTrain["WAIT_TIME"] = fareTrain["WAIT_TIME"]*2
fareTrain["TOTAL_LUGGAGE_WEIGHT"] = fareTrain["TOTAL_LUGGAGE_WEIGHT"]*2
fareTrain["STARTING_LATITUDE"] = fareTrain["STARTING_LATITUDE"]*2
fareTrain["DESTINATION_LATITUDE"] = fareTrain["DESTINATION_LATITUDE"]*2
         
   
''' 
 
for dataset in fullData:
    dataset['TOTAL_LUGGAGE_WEIGHT'] = dataset['TOTAL_LUGGAGE_WEIGHT'].fillna(0)
    dataset['WAIT_TIME'] = dataset['WAIT_TIME'].fillna(0)
    dataset['STARTING_LATITUDE'] = dataset['STARTING_LATITUDE'].fillna(0)
    dataset['STARTING_LONGITUDE'] = dataset['STARTING_LONGITUDE'].fillna(0)
    dataset.loc[dataset['STARTING_LATITUDE'].isnull(),'STARTING_LATITUDE'] = dataset['DESTINATION_LATITUDE']
    dataset.loc[dataset['STARTING_LONGITUDE'].isnull(),'STARTING_LONGITUDE'] = dataset['DESTINATION_LONGITUDE']
    
    
    


    
#print list(set(fareTrain.VEHICLE_TYPE))
#print fareTrain.describe(include='all')
#print fareTrain.info()
'''
for col in fareTrain:
    print (type(fareTrain[col][1]))
'''


fareTrain["TIMESTAMP"] = pd.to_datetime(fareTrain["TIMESTAMP"] )
fareTest["TIMESTAMP"] = pd.to_datetime(fareTest["TIMESTAMP"] )

'''for col in fareTrain:
    print (type(fareTrain[col][1]))
'''


column_1 = fareTrain["TIMESTAMP"]
data1 = pd.DataFrame({
    #        "year": column_1.dt.year,
             "month" :column_1.dt.month,
      #      "day":column_1.dt.day,
             "hour":column_1.dt.hour,
  #           "minute" :column_1.dt.minute,
  #           "second":column_1.dt.second,
             })
#fareTrain = fareTrain.append(data1)
fareTrain = pd.concat([fareTrain, data1], axis=1)
    
    
column_1 = fareTest["TIMESTAMP"]
data1 = pd.DataFrame({
        #     "year": column_1.dt.year,
             "month" :column_1.dt.month,
         #    "day":column_1.dt.day,
             "hour":column_1.dt.hour,
   #          "minute" :column_1.dt.minute,
    #         "second":column_1.dt.second,
             })
#fareTrain = fareTrain.append(data1)
fareTest = pd.concat([fareTest, data1], axis=1)
   
 



fareTrain = fareTrain.drop("TIMESTAMP",axis=1)
fareTest = fareTest.drop("TIMESTAMP",axis=1)

#fareTrain["new"] = fareTrain["TOTAL_LUGGAGE_WEIGHT"]*fareTrain["WAIT_TIME"]*fareTrain["VEHICLE_TYPE"]
#fareTest["new"] = fareTest["TOTAL_LUGGAGE_WEIGHT"]*fareTest["WAIT_TIME"]*fareTest["VEHICLE_TYPE"]

dropElements = ["ID","DESTINATION_LONGITUDE","DESTINATION_LATITUDE"]
fareTrain = fareTrain.drop(dropElements,axis = 1)
fareTest = fareTest.drop(dropElements,axis = 1)
#print fareTrain.info()

#print fareTrain.describe()
'''
avj1 = 20.111445
avj2 =  20.129680   
avj3 = 79.794135
avj4 =  79.793273
davj = 0.0321
   
'''           
      
#fareTrain = fareTrain[np.isfinite(fareTrain["STARTING_LATITUDE"])]
#fareTrain = fareTrain[np.isfinite(fareTrain["STARTING_LONGITUDE"])]
#fareTrain = fareTrain[np.isfinite(fareTrain["DESTINATION_LATITUDE"])]
#fareTrain = fareTrain[np.isfinite(fareTrain["DESTINATION_LONGITUDE"])]
#print fareTest.info()   
#print fareTrain.info()           


#sns.distplot(fareTrain['FARE']);
'''
var = 'VEHICLE_TYPE'
data = pd.concat([fareTrain['FARE'], fareTrain[var]], axis=1)
data.plot.scatter(x=var, y='FARE', ylim=(0,2000));
'''
'''
sns.set(style="ticks", color_codes=True)
g = sns.pairplot(fareTrain, dropna = True)
'''

'''
sns.set(style="ticks", color_codes=True)
grid = sns.PairGrid(fareTrain)
grid.map( plt.scatter )
#sns.plt.show()
'''

'''
colormap = plt.cm.viridis
plt.figure(figsize = (12,12))
plt.title('Correlation of Features', y = 1.05, size = 15)
sns.heatmap(fareTrain.corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
'''
'''
var = 'Location'
data = pd.concat([TrainData['Golden Grains'], TrainData[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="Golden Grains", data=data)
fig.axis(ymin=500000, ymax=1000000);
'''
fareTrain1 = fareTrain[:12000]
fareTrain2 = fareTrain[12000:]

#fareTrain = fareTrain.sort_values("WAIT_TIME")
#fareTest = fareTest.sort_values("WAIT_TIME")

#print fareTrain.head()
#print fareTest.head()

Y_train = fareTrain["FARE"]
Y_test = fareTrain2["FARE"]
X_train = fareTrain.drop("FARE",axis = 1)
X_test = fareTrain2.drop("FARE",axis = 1)



def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, X_train, Y_train, scoring="neg_mean_squared_error", cv = 5))
    return(rmse)

'''
model_ridge = Ridge()
alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]
cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean() 
            for alpha in alphas]

cv_ridge = pd.Series(cv_ridge, index = alphas)
print cv_ridge.min()

'''


'''
model_lasso = LassoCV(alpha=1).fit(X_train, Y_train)

print rmse_cv(model_lasso).mean()


coef = pd.Series(model_lasso.coef_, index = X_train.columns)
imp_coef = pd.concat([coef.sort_values().head(10),
                     coef.sort_values().tail(10)])

matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title("Coefficients in the Lasso Model")
  
'''

'''
#sns.distplot(fareTrain['FARE'], fit=norm);
fig = plt.figure()
res = stats.probplot(fareTrain['FARE'], plot=plt)
'''



'''

degree = 4
model = make_pipeline(PolynomialFeatures(degree),LinearRegression())
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)

print r2_score(Y_test,Y_pred)*200
#print rmse_cv(model).mean()
#plt.plot(Y_pred, Y_test, color="gold", linewidth=lw,
        #label="degree %d" % degree)
'''

model = ensemble.GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=3, max_features='sqrt',
                                               min_samples_leaf=15, min_samples_split=10, loss='huber')

model.fit(X_train, Y_train)
Y_pred = model.predict(fareTest)
#print r2_score(Y_test,Y_pred)*200

#scores = cross_val_score(model, X_train,Y_train , cv=10, scoring='r2')
#print scores


#print (Y_pred)[:20]
#print (Y_test)[:20]
'''
lm = LinearRegression()
model = lm.fit(X_train,Y_train)
Y_pred = model.predict(X_test)

'''

#print (Y_pred)[:10]
#print (Y_test)[:10]
StackingSubmission = pd.DataFrame({"ID":id1,"FARE": Y_pred })
StackingSubmission["ID"] = StackingSubmission["ID"]
StackingSubmission["FARE"] = StackingSubmission["FARE"]
StackingSubmission = StackingSubmission[["ID","FARE"]]
StackingSubmission.to_csv("StackingSubmission.csv", index=False)



