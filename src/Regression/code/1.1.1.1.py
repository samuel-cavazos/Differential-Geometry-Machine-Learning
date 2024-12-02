#!pip install matplotlib
#!pip install numpy
#!pip install sklearn

def f(x):
    return x**2 + 2*x + 1

# Plot using matplotlib
import matplotlib.pyplot as plt
import numpy as np
import random

# Define the true function
def f(x):
    return x**2 + 2*x + 1

# Generate data
np.random.seed(42)
x = np.linspace(0, 10, 200)
y_true = f(x)
y_data = y_true + np.random.uniform(-10, 10, size=x.shape)

# split into training and test datasets (80% training, 20% test) 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y_data, test_size=0.2)
print(f'Train size: {len(x_train)}')
print(f'Test size: {len(x_test)}')

# Plot the data and the true function, coloring the training and test data differently
plt.plot(x, y_true,color='black', label='True function')
plt.scatter(x_train, y_train, color='darkred', label='Training data')
plt.scatter(x_test, y_test, color='blue', label='Test data')
plt.legend()
plt.savefig('fig1.png')
plt.show()
