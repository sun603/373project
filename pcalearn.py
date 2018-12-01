# Input: number of features F
# numpy matrix X, with n rows (samples), d columns (features)
# Output: numpy vector mu, with d rows, 1 column
# numpy matrix Z, with d rows, F columns
import numpy as np
import numpy.linalg as la
def run(F,X):
    n,d = X.shape
    mu = np.zeros(d)
    for i in range(d):
        mu[i] = 1.0/n * np.sum([X[t][i] for t in range(n)])
    for t in range(n):
        for i in range(d):
            X[t][i] = X[t][i] - mu[i]

    U,s,Vt = la.svd(X,False)
    g = s[0:F]
    for i in range(F):
        if g[i] > 0:
            g[i] = 1.0 / g[i]

    W = Vt[0:F]
    Z = np.dot(W.T,np.diag(g) )
    mu = np.reshape(mu,(d,1))
    return (mu, Z)