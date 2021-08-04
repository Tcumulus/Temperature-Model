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

best, best_b = 100000, 0
total_best, total_b = 100000, 0
best_l = []
def train(b):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1) # Train and test data chunks
    linear = linear_model.LinearRegression()
    linear = linear.fit(x_train, y_train)

    y_test = list(y_test)
    x_pred = list()
    for row in x_test:
        predict = float(list(linear.predict([row]))[0])
        x_pred.append(predict * b)

    acc_l = list()
    for n in range(len(x_pred)):
        acc_l.append(abs(x_pred[n] - y_test[n]))
    accuracy = sum(acc_l) / len(acc_l)

    global best
    global best_b
    if accuracy < best:
        best = accuracy
        best_b = b

def b_loop():
    global best, best_b, total_best, total_b, best_l, best_b_l
    best, best_b = 100000, 0
    [train((b+1)/10) for b in range(50)] #Train

    best_l.append(best_b)
    if best < total_best:
        total_best = best
        total_b = best_b
    print(best, best_b)

[b_loop() for _ in range(25)]

print(f" -- {total_best} -- {total_b} -- {sum(best_l) / len(best_l)}")