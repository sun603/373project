import numpy as np
import matplotlib.pyplot as pp
from sklearn import svm

# reading the original format of the data
X = np.loadtxt('datatraining.txt', delimiter=',', skiprows=1, usecols=range(2,7))
y = np.loadtxt('datatraining.txt', delimiter=',', skiprows=1, usecols=range(7,8))
X = X.reshape((-1,5))
y = y.reshape((-1,1))

# convert 0 to -1 
samples_with_two = list(np.where(y==0)[0])
y[samples_with_two] = -1

p_sample = list(np.where(y==1)[0])
n_sample = list(np.where(y==-1)[0])

# visualize feature 1 and 4
pp.figure()
pp.plot(m, 'b+')
pp.xlabel('temp')
pp.ylabel('mean')


pp.show()


# change matrix to array
y = np.squeeze(np.asarray(y))

# perform svm on training data
clf.fit(X, y) 

# make prediction with testing data
clf.predict([[23.718,26.29,578.4,760.4,0.00477266099212519]])
  # Output: array([1.])
clf.predict([[22.39,25,0,805.5,0.0041841297156888]])
  # Output: array([-1.])
