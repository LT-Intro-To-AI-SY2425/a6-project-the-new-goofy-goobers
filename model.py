"https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

"creating model"

data = pd.read_csv("data.csv")
# x = data[["x1", "x2", "x3"]].values    PLACEHOLDER REPLACE WITH ACTUAL DATA HEADERS
# y = data["y"].values                   SAME HERE

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = .2)

# # reshape the xtrain data into a 2D array
# xtrain = xtrain.reshape(-1, 1)         I FORGOT WHAT THIS PART DOES ILL FIGURE OUT IF WE NEED IT LATER

model = LinearRegression().fit(xtrain, ytrain)

coef = np.around(model.coef_, 2)
intercept = round(float(model.intercept_), 2)
r_squared = round(model.score(x, y),2)

print(f"linear equation: y={coef[0]}x1 + {coef[1]}x2 + {coef[2]}x3 + {intercept}")
print("r^2:", r_squared)

"testing"

# get the predicted y values for the xtest values - returns an array of the results
predict = model.predict(xtest)
# round the value in the np array to 2 decimal places
predict = np.around(predict, 2)
print(predict)

# compare the actual and predicted values
print("\nTHE TESTING")
for index in range(len(xtest)):
    actual = ytest[index] # gets the actual y value from the ytest dataset
    predicted_y = predict[index] # gets the predicted y value from the predict variable
    x_coord = xtest[index] # gets the x value from the xtest dataset
    print(f"x1: {x_coord[0]} x2: {x_coord[1]} x3: {x_coord[2]} actual: {actual} predicted: {predicted_y}")