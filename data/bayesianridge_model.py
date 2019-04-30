import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pickle
from sklearn.linear_model import BayesianRidge





df = pd.read_csv('Admission_Predict_Ver1.1.csv')
df.rename(columns = {'Chance of Admit ':'Chance of Admit', 'LOR ':'LOR'}, inplace=True)
df.drop(labels='Serial No.', axis=1, inplace=True)

y = df["Chance of Admit"].values
x = df.drop(["Chance of Admit"],axis=1)

# separating train (80%) and test (%20) sets
x_train, x_test,y_train, y_test = train_test_split(x,y,test_size = 0.20,random_state = 42)


scalerX = MinMaxScaler(feature_range=(0, 1))
x_train[x_train.columns] = scalerX.fit_transform(x_train[x_train.columns])
x_test[x_test.columns] = scalerX.transform(x_test[x_test.columns])

br = BayesianRidge()
br.fit(x_train,y_train)

with open('bayesianridge_model.pkl', 'wb') as file:
    pickle.dump(br, file)