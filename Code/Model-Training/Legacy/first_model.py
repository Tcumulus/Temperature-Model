import pandas as pd
import numpy as np
from sklearn import linear_model
import sklearn
import pickle
import matplotlib.pyplot as plt

data = pd.read_csv("/Users/Maarten/Documents/Projects/Temperature-Model/Data/Final_Data.txt", sep=",", low_memory=False) # import
data.dropna(inplace=True)
data["TC"] = data["TC"].abs()

predict = "TC"
x = np.array(data.drop(["MM", "HH", predict], axis=1))
y = np.array(data[predict])
x.astype(float)
y.astype(float)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1) # Train and test data chunks
linear = linear_model.LinearRegression() 

linear.fit(x_train,y_train)
accuracy = linear.score(x_test, y_test)
print(accuracy)

p = "N"
plt.scatter(data[p], data["TC"])
plt.xlabel(p)
plt.ylabel("TC")
plt.show()