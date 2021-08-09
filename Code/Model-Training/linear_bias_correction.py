import random as rd
from numpy.lib.function_base import append
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

import sklearn
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split 

# init
best, total_best, best_b, total_b= 100000, 100000, 0, 0
best_l, temp_prediction, avg_b, avg_temp_prediction, linear, append_row = [], [], [], [], [], []

# input
month, hour = int(input("Month: ")), int(input("Hour: ")) - 2 # next hour
current_temperature = float(input("Temperature (°C): "))
current_wind_speed = float(input("Wind speed (km/h): ")) / 3.6 * 10 # convert to 0.1m/s
current_wind_dir = int(input("Wind direction (°): "))
current_cloud_cover = int(input("Cloud cover (0-8): "))
prediction_hours = 10
members = 10 #int(input("Members (#): "))
gas = 10

print(f"ET: {gas*prediction_hours*0.15}s")
if hour <= 0: hour += 24

# output preparation
avg_temp_prediction.append(current_temperature)
temp_prediction = np.empty([prediction_hours+1, members+1])
temp_prediction[0] = np.array([current_temperature] * (members+1))
running_temperature = [current_temperature * 10] * (members + 1)
linear = []

# bias training
def train(b, i, x, y):
    global linear, best, best_b

    l = linear_model.LinearRegression()
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1) # Train and test data chunks
    l = l.fit(x_train, y_train)

    y_test, x_pred, acc_l = list(y_test), list(), list()

    [x_pred.append(float(list(l.predict([row]))[0]) * b) for row in x_test]
    [acc_l.append(abs(x_pred[n] - y_test[n])) for n in range(len(x_pred))]
    accuracy = sum(acc_l) / len(acc_l)

    if accuracy < best:
        best, best_b = accuracy, b
        with open("/Users/Maarten/Documents/Projects/Temperature-Model/Data/temp_model.pickle", "wb") as f:
                pickle.dump(l, f)

# multiple assurance
def b_loop(i, gas, hour):
    lookup_hour = hour + i
    if lookup_hour > 24: lookup_hour -= 24
    data = pd.read_csv(f"/Users/Maarten/Documents/Projects/Temperature-Model/Data/Final_data/{month}_{lookup_hour}.txt", sep=",")
    print(f"00.+{i}: opening /{month}_{lookup_hour}.txt")
    data.dropna(inplace=True)

    # hardcode input
    predict = "TC"
    x = np.array(data[["DD", "FF", "N"]])
    y = np.array(data[predict])
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)

    global best, best_b, total_best, total_b, best_l
    for _ in range(gas):
        best, best_b = 100000, 0
        [train((b+1)/10, i, x, y) for b in range(10)] #Train
        best_l.append(best_b)
        if best < total_best: total_best, total_b = best, best_b

    load_model = open("/Users/Maarten/Documents/Projects/Temperature-Model/Data/temp_model.pickle", "rb") # Open existing model
    linear.append(pickle.load(load_model)) 
    avg_b.append(sum(best_l) / len(best_l))
    print(f"Bias correction, b: {round(avg_b[i], 2)}")

# general run
def run(member, hour):
    global current_wind_dir, current_wind_speed, current_cloud_cover, running_temperature, avg_b, append_row 
    append_row.append(float((linear[hour].predict([[current_wind_dir, current_wind_speed, current_cloud_cover]])) * avg_b[hour] + running_temperature[member]) / 10) # prediction


# general training loop
print(f"00: Start training for {gas} gas and {prediction_hours} frames")
[b_loop(i, gas, hour) for i in range(prediction_hours)]

for n in range(prediction_hours):
    append_row = []

    # oper
    print(f"01.+{n+1} Running")
    run(0, n)
    o_wind_dir, o_wind_speed, o_cloud_cover, o_b = current_wind_dir, current_wind_speed, current_cloud_cover, avg_b

    # members
    for member in range(members):
        current_wind_dir = o_wind_dir + rd.randint(-72, 72)
        current_wind_speed = o_wind_speed + rd.randint(-50, 50) 
        current_cloud_cover = o_cloud_cover + rd.randint(-2, 2)
        random_factor = rd.uniform(-0.25, 0.25)
        avg_b = [x + random_factor for x in o_b] 

        if current_wind_dir < 0: current_wind_dir = 0
        if current_wind_speed < 0: current_wind_speed = 0
        if current_cloud_cover < 0: current_cloud_cover = 0
        elif current_cloud_cover > 8: current_cloud_cover = 8
        run(member+1, n)

    temp_prediction[n+1] = np.array(append_row)

    running_temperature = [x*10 for x in append_row]
    avg_temp_prediction.append(sum(append_row) / len(append_row))

print("02: Finished")

# plot
h = []
for i in range(prediction_hours+1):
    x_hour = hour + i + 2
    #if x_hour > 24: x_hour -= 24
    h.append(x_hour) # with two for utc time differce

oper = temp_prediction[:, 0]
np.delete(temp_prediction, [0], axis=1)
[plt.plot(h, column, "#c7c5bf") for column in temp_prediction.T]
plt.plot(h, oper, "k", label = "P0")
plt.plot(h, avg_temp_prediction, "r", label = "avg")
plt.legend()
plt.xlabel("Time (h UTC+2)")
plt.ylabel("Temperature (°C)")
plt.show()