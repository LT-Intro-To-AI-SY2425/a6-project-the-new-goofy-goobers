"https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
"make da model"
data = pd.read_csv("data.csv")


"RAHHHHHHHHHH"
# x = data[["x1", "x2", "x3"]].values    PLACEHOLDER REPLACE WITH ACTUAL DATA HEADERS
# y = data["y"].values                   SAME HERE



xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=.2)
xtrain=xtrain.reshape(-1,1)
model=LinearRegression().fit(xtrain,ytrain)
coef=np.around(model.coef_,2)
intercept=round(float(model.intercept_),2)
r2=round(model.score(x,y),2)
print(f"equation: y={coef[0]}x1 + {coef[1]}x2 + {coef[2]}x3 + {intercept}")
print("r^2:",r2)
"testing"
predict=model.predict(xtest)
predict=np.around(predict,2)
print(predict)
print("\nTHE TESTING")
for index in range(len(xtest)):
    actual=ytest[index]
    predicted_y=predict[index]
    x_coord=xtest[index]
    print(f"x1: {x_coord[0]} x2: {x_coord[1]} x3: {x_coord[2]} actual: {actual} predicted: {predicted_y}")