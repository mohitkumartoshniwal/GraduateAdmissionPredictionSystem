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
from sklearn import preprocessing
#from sklearn.cross_validation import cross_val_score, cross_val_predict




df = pd.read_csv('Admission_Predict_Ver1.1.csv')
df.rename(columns = {'Chance of Admit ':'Chance of Admit', 'LOR ':'LOR'}, inplace=True)
df.drop(labels='Serial No.', axis=1, inplace=True)

y = df["Chance of Admit"].values
x = df.drop(["Chance of Admit"],axis=1)

X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=10)

scalerX = MinMaxScaler(feature_range=(0, 1))

X_train[X_train.columns] = scalerX.fit_transform(X_train[X_train.columns])
X_test[X_test.columns] = scalerX.transform(X_test[X_test.columns])



lm = LinearRegression()
model = lm.fit(X_train, y_train)
predictions = lm.predict(X_test)




#cross_val_score(lm, x, y, cv=3)



with open('LinearRegression.pkl', 'wb') as file:
    pickle.dump(lm, file)