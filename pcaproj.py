# Input: number of features F
# numpy matrix X, with n rows (samples), d columns (features)
# numpy vector mu, with d rows, 1 column
# numpy matrix Z, with d rows, F columns
# Output: numpy matrix P, with n rows, F columns
import numpy as np
import numpy.linalg as la
def run(X,mu,Z):
    # Your code goes here
    n,d = X.shape
    for t in range(n):
        for i in range(d):
            X[t][i] = X[t][i] - mu[i]
    P = np.dot(X,Z)
    return P