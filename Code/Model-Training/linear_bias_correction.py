import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split 

linear = linear_model.LinearRegression()
best, total_best, best_b, total_b = 100000, 100000, 0, 0
best_l = []

# bias training
def train(b):
    global linear, best, best_b

    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1) # Train and test data chunks
    linear = linear.fit(x_train, y_train)
    y_test, x_pred, acc_l = list(y_test), list(), list()

    [x_pred.append(float(list(linear.predict([row]))[0]) * b) for row in x_test]
    [acc_l.append(abs(x_pred[n] - y_test[n])) for n in range(len(x_pred))]
    accuracy = sum(acc_l) / len(acc_l)

    if accuracy < best: best, best_b = accuracy, b

# multiple assurance
def b_loop():
    global best, best_b, total_best, total_b, best_l
    best, best_b = 100000, 0
    [train((b+1)/10) for b in range(25)] #Train

    best_l.append(best_b)
    if best < total_best: total_best, total_b = best, best_b

# input
month, hour = int(input("Month: ")), int(input("Hour: "))
current_temperature = int(input("Temperature (°C): "))
current_wind_speed = int(input("Wind speed (km/h): ")) * 0.36 # convert to 0.1m/s
current_wind_dir = int(input("Wind direction (°): "))
current_cloud_cover = int(input("Cloud cover (0-8): "))
gas = 10

# import
data = pd.read_csv(f"/Users/Maarten/Documents/Projects/Temperature-Model/Data/Final_data/{month}_{hour}.txt", sep=",")
data.dropna(inplace=True)

# hardcode input
predict = "TC"
x = np.array(data[["DD", "FF", "N"]])
y = np.array(data[predict])
x = np.array(x, dtype=float)
y = np.array(y, dtype=float)

# general loop
[b_loop() for _ in range(gas)]
avg_b = (sum(best_l) / len(best_l)) / 10

# prediction and output
temp_prediction = float(linear.predict([[current_wind_dir, current_wind_speed, current_cloud_cover]])) * avg_b
temp_prediction += current_temperature
print(f"{hour+1}h: {round(temp_prediction, 1)}°C")