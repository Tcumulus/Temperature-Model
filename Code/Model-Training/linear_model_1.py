import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split 
import sklearn
import pickle
import matplotlib.pyplot as plt

month = int(input("Month: "))
hour = int(input("Hour: "))

data = pd.read_csv(f"/Users/Maarten/Documents/Projects/Temperature-Model/Data/Final_data/{month}_{hour}.txt", sep=",")
data.dropna(inplace=True)

predict = "DD"
x = np.array(data.drop([predict], axis=1))
y = np.array(data[predict])
x.astype(float)
y.astype(float)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1) # Train and test data chunks
linear = linear_model.LinearRegression()
linear = linear.fit(x_train, y_train)
y_pred = linear.predict(x_test)

accuracy = linear.score(x_test, y_test)
print(accuracy)