from numpy.lib.function_base import average
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split 
import sklearn
import pickle

month = int(input("Month: "))
hour = int(input("Hour: "))

data = pd.read_csv(f"/Users/Maarten/Documents/Projects/Temperature-Model/Data/Final_data/{month}_{hour}.txt", sep=",")
data.dropna(inplace=True)

predict = "TC"
x = np.array(data[["DD", "FF", "N"]]) # x = np.array(data.drop([predict], axis=1))
y = np.array(data[predict])
x = np.array(x, dtype=float)
y = np.array(y, dtype=float)

linear = linear_model.LinearRegression()

best = 0
def train():
    global linear
    global best

    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1) # Train and test data chunks
    linear = linear.fit(x_train, y_train)

    accuracy = linear.score(x_test, y_test)
    print(accuracy)

    if accuracy < best:
        best = accuracy

[train() for _ in range(25)]

print(f" -- {round(best, 2)} -- ")
print(f"{round(float(linear.predict([[0,0,0]]))/10, 1)}°C")