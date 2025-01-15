import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = pd.read_csv("data.csv")

x_1 = data["Displaced"]
x_2 = data["Debtor"]
x_3 = data["Scholarship holder"]
x_4 = data["Gender"]


y_raw=data["Target"].values
y=[]
for status in y_raw:
    if status == "Dropout":
        y.append(0)
    elif status == "Enrolled":
        y.append(0.5)
    elif status == "Graduate":
        y.append(1)

fig, graph = plt.subplots(4)

# Models Shown

graph[0].scatter(x_1, y)
graph[0].set_xlabel("Displaced")
graph[0].set_ylabel("Target")

graph[1].scatter(x_2, y)
graph[1].set_xlabel("Debtor")
graph[1].set_ylabel("Target")

graph[2].scatter(x_3, y)
graph[2].set_xlabel("Scholarship holder")
graph[2].set_ylabel("Target")

graph[3].scatter(x_4, y)
graph[3].set_xlabel("Gender")
graph[3].set_ylabel("Target")

y = pd.Series(y)
print(f"Correlation between Displaced and Target: {round(x_1.corr(y), 3)}")
print(f"Correlation between Debtor and Target: {round(x_2.corr(y), 3)}")
print(f"Correlation between Scholarship Holder and Target: {round(x_3.corr(y), 3)}")
print(f"Correlation between Gender and Target: {round(x_4.corr(y), 3)}")

plt.tight_layout()
plt.show()