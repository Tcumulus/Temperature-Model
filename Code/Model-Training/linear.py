import random as rd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sklearn
from sklearn import linear_model
import sklearn.metrics as sm

temp_prediction, avg_b, avg_temp_prediction, linear, append_row = [], [], [], [], []


# input
prediction_hours = 5
members = 10
gas = 100
utc = 2
month, hour = int(input("Month: ")), int(input("Hour: ")) - utc  # next hour
current_temperature = float(input("Temperature (°C): "))
current_wind_speed = float(
    input("Wind speed (km/h): ")) / 3.6 * 10  # in 0.1m/s
current_wind_dir = int(input("Wind direction (°): "))
current_cloud_cover = int(input("Cloud cover (0-8): "))

if hour <= 0:
    hour += 24

# output preparation
avg_temp_prediction.append(current_temperature)
temp_prediction = np.empty([prediction_hours+1, members+1])
temp_prediction[0] = np.array([current_temperature] * (members+1))
running_temperature = [current_temperature * 10] * (members + 1)
linear = []


def train(i):
    global linear, gas

    lookup_hour = hour + i
    if lookup_hour > 24:
        lookup_hour -= 24
    data = pd.read_csv(
        f"/Users/Maarten/Documents/Projects/Temperature-Model/Data/Final_data/{month}_{lookup_hour}.txt", sep=",")
    print(f"00.+{i}: opening /{month}_{lookup_hour}.txt")
    data.dropna(inplace=True)

    # hardcode input
    predict = "TC"
    if o_wind_speed < 2:
        x = np.array(data[["FF", "N"]])
    else:
        x = np.array(data[["DD", "FF", "N"]])
    y = np.array(data[predict])
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)

    best_l = linear_model.LinearRegression()
    best = 1
    for _ in range(gas):
        l = linear_model.LinearRegression()
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
            x, y, test_size=0.3)
        l = l.fit(x_train, y_train)

        y_pred = l.predict(x_test)
        accuracy = sm.r2_score(y_test, y_pred)

        if accuracy < best:
            best = accuracy
            best_l = l

    print(
        f"01.Finished training with resulting best accuracy of {round(best,2)}")
    linear.append(best_l)


def run(member, hour):
    global current_wind_dir, current_wind_speed, current_cloud_cover, running_temperature, append_row
    if o_wind_speed < 2:
        append_row.append(float((linear[hour].predict(
            [[current_wind_speed, current_cloud_cover]])) + running_temperature[member]) / 10)
    else:
        append_row.append(float((linear[hour].predict(
            [[current_wind_dir, current_wind_speed, current_cloud_cover]])) + running_temperature[member]) / 10)


# general training loop
print(f"00.Start training for {gas} gas and {prediction_hours} frames")
[train(i) for i in range(prediction_hours)]

for n in range(prediction_hours):
    append_row = []

    # oper
    print(f"02.+{n+1} Running")
    run(0, n)
    o_wind_dir, o_wind_speed, o_cloud_cover = current_wind_dir, current_wind_speed, current_cloud_cover

    # members
    for member in range(members):
        current_wind_dir = o_wind_dir + rd.randint(-45, 45)
        current_wind_speed = o_wind_speed + rd.randint(-50, 50)
        current_cloud_cover = o_cloud_cover + rd.randint(-2, 2)

        current_wind_dir = 360 - current_wind_dir if current_wind_dir < 0 else current_wind_dir
        current_wind_speed = 0 if current_wind_speed < 0 else current_wind_speed
        current_cloud_cover = 0 if current_cloud_cover < 0 else current_cloud_cover
        current_cloud_cover = 8 if current_cloud_cover > 8 else current_cloud_cover

        run(member+1, n)

    temp_prediction[n+1] = np.array(append_row)

    running_temperature = [x*10 for x in append_row]
    avg_temp_prediction.append(sum(append_row) / len(append_row))

print("03.Finished")

# plot
h = []
for i in range(prediction_hours+1):
    x_hour = hour + i + utc
    if x_hour > 24:
        x_hour -= 24
    h.append(str(x_hour))

oper = temp_prediction[:, 0]
np.delete(temp_prediction, [0], axis=1)
[plt.plot(h, column, "#c7c5bf") for column in temp_prediction.T]
plt.plot(h, oper, "k", label="P0")
plt.plot(h, avg_temp_prediction, "r", label="avg")
plt.xticks(h)
plt.legend()
plt.xlabel(f"Time (h UTC+{utc})")
plt.ylabel("Temperature (°C)")
plt.show()
