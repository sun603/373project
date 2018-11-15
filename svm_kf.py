
import numpy as np
import matplotlib as plt
import pickle
#set backend for matplotlib
plt.use('Agg')
import matplotlib.pyplot as pp
#install sklearn first : using command "pip install -U scikit-learn"
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_gaussian_quantiles
k = 5

X,Y = pickle.load(open("kf.p","rb"))
Xtest,ytest = pickle.load(open("testset.p","rb"))

# print(Y[1][101])

accuracy_set = []
for i in range(k):
    X_train = np.concatenate([x for xi,x in enumerate(X) if xi!= i])
    y_train = np.concatenate([y for yi,y in enumerate(Y) if yi!= i])
    clf = svm.SVC(gamma='scale')
    clf.fit(X_train, y_train)
    (n, dx) = X[i].shape
    correct = 0.0
    for j in range(n):
        pred = clf.predict(X[i][j].reshape((1,5)))
        if Y[i][j] == pred:
            correct += 1.0
    accuracy = correct/n
    accuracy_set.append(accuracy)


# accuracy_mean = 1.0 / len(accuracy_set) * sum(accuracy_set)
# print accuracy_mean
accuracy_mean = np.mean(accuracy_set)
print accuracy_mean
accuracy_var = np.var(accuracy_set)
print accuracy_var

