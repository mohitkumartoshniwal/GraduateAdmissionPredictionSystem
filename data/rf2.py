import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pickle
from sklearn.linear_model import LinearRegression





df = pd.read_csv('Admission_Predict_Ver1.1.csv')
df.rename(columns = {'Chance of Admit ':'Chance of Admit', 'LOR ':'LOR'}, inplace=True)
df.drop(labels='Serial No.', axis=1, inplace=True)



from sklearn import preprocessing
minmax_scaler = preprocessing.MinMaxScaler()
minmax_scaler_fit=minmax_scaler.fit(df[['GRE Score', 'TOEFL Score']])
NormalizedGREScoreAndTOEFLScore = minmax_scaler_fit.transform(df[['GRE Score', 'TOEFL Score']])


# Creating a separate Data Frame just to store new standardized columns
NormalizedGREScoreAndTOEFLScoreData=pd.DataFrame(NormalizedGREScoreAndTOEFLScore,columns=['GRE Score', 'TOEFL Score'])
NormalizedGREScoreAndTOEFLScoreData.head()

df['GRE Score']=NormalizedGREScoreAndTOEFLScoreData['GRE Score']
df['TOEFL Score']=NormalizedGREScoreAndTOEFLScoreData['TOEFL Score']

PredictorColumns=list(df.columns)
PredictorColumns.remove('Chance of Admit')

X=df[PredictorColumns].values
y=df['Chance of Admit'].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=10)

from sklearn.ensemble import RandomForestRegressor
clf = RandomForestRegressor(n_estimators=100,criterion='mse')
#print(clf)
RF=clf.fit(X_train,y_train)



with open('RandomForestRegressor.pkl', 'wb') as file:
    pickle.dump(clf, file)