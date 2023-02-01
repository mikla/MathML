# First NN 

import numpy as np
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

np.random.seed(3) 

m = 30

X, Y = make_regression(n_samples=m, n_features=1, noise=20, random_state=1)

print('Regression data X:')
print(X)
print('Regression data Y')
print(Y)

X = X.reshape((1, m))
Y = Y.reshape((1, m))

print('Training dataset X:')
print(X)
print('Training dataset Y')
print(Y)

plt.scatter(X,  Y, c="black")

plt.xlabel("$x$")
plt.ylabel("$y$")

plt.show()
