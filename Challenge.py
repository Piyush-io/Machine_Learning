import pandas as pd
import numpy as np

data = pd.read_csv("/Users/bruh/Downloads/Experience.csv")
x_values, y_values = data['x'].values, data['y'].values
m, b = 0.0, 0.0
learning_rate, iterations = 0.001, 100_000

for i in range(iterations):
    predictions = m * x_values + b
    error = predictions - y_values
    m -= learning_rate * 2 * np.dot(error, x_values) / len(data)
    b -= learning_rate * 2 * np.sum(error) / len(data)

def predict(x):
    return m * x + b

print(m,b)

while True:
        experience = float(input("Enter employees' experience: "))
        print("Predicted salary:", predict(experience))