import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier 
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

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1) # Train and test data chunks
tree = DecisionTreeClassifier()
tree = tree.fit(x_train, y_train)
y_pred = tree.predict(x_test)

accuracy = tree.score(x_test, y_test)
print(accuracy)

predictions = tree.predict([[0,100,8]])
print(predictions)

#predictions = tree.predict(x_test)

#for x in range(len(predictions)):
#    print(predictions[x], y_test[x])