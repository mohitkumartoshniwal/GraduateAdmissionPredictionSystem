import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys
import os
df = pd.read_csv('Admission_P.csv')
print("There are",len(df.columns),"columns:")
for x in df.columns:
    sys.stdout.write(str(x)+", ")
    df=df.rename(columns = {'Chance of Admit ':'Chance of Admit'})
  
    serialNo = df["Serial No."].values

df.drop(["Serial No."],axis=1,inplace = True)
y = df["Chance of Admit"].values
x = df.drop(["Chance of Admit"],axis=1)

# separating train (80%) and test (%20) sets
from sklearn.model_selection import train_test_split

x_train, x_test,y_train, y_test = train_test_split(x,y,test_size = 0.20,random_state = 42)
corr_matrix = df.corr()
corr_matrix["Chance of Admit"].sort_values(ascending=False)
from sklearn.preprocessing import MinMaxScaler
scalerX = MinMaxScaler(feature_range=(0, 1))
x_train[x_train.columns] = scalerX.fit_transform(x_train[x_train.columns])
x_test[x_test.columns] = scalerX.transform(x_test[x_test.columns])
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train,y_train)


# save the model to disk
filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))




y_head_lr = lr.predict(x_test)

print("Predicted values for some input")
print("real value of y_test[1]: " + str(y_test[1]) + " -> the predict: " + str(lr.predict(x_test.iloc[[1],:])))
print("real value of y_test[2]: " + str(y_test[2]) + " -> the predict: " + str(lr.predict(x_test.iloc[[2],:])))

from sklearn.metrics import r2_score
print("r_square score: ", r2_score(y_test,y_head_lr))

y_head_lr_train = lr.predict(x_train)
print("r_square score (train dataset): ", r2_score(y_train,y_head_lr_train))
result = r2_score(y_train,y_head_lr_train)



loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(x_train, y_train)
print(result)
from sklearn.externals import joblib
joblib.dump(linear, 'linear_model.pkl')
linear_model = joblib.load('linear_model.pkl')
ex2 = np.array([6.2,3.4,5.4,2.3]).reshape(1,-1)
linear_model.predict(ex2)