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
#k = 5
def getAdaBoostAccuracy(X_train, y_train, rate):
	bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),learning_rate = rate)
	bdt.fit(X_train, y_train)
	(n, dx) = X[i].shape
	correct = 0.0
	for j in range(n):
		pred = bdt.predict(XTest[i].reshape((1,5)));
		if Y[i][j] == pred:
			correct += 1.0
	accuracy = correct/n
	return accuracy
def inter(X_train, y_train, rate):
	return getAdaBoostAccuracy(X_train,y_train, rate)
	

d1 = np.loadtxt("datatest.txt",
    delimiter=',',skiprows=1,usecols=range(2,8))

d2 = np.loadtxt("datatest2.txt",
    delimiter=',',skiprows=1,usecols=range(2,8))

d3 = np.loadtxt("datatraining.txt",
    delimiter=',',skiprows=1,usecols=range(2,8))

#dall = np.concatenate([d1, d2, d3])
dall = np.array(d3);
np.random.shuffle(dall)


k = 5
pos = int(len(dall) * 0.3)
pos += (len(dall) - pos) %k
testset = dall[:pos]
training = dall[pos:]
kf = np.split(training,k)
X = [ [] for i in range(k)]
Y = [ [] for i in range(k)]
for i in range(k):
	X[i] = kf[i][:,0:5]
	Y[i] = kf[i][:,5]
	# convert 0 to -1 for both training and testing dataset
	training_with_two = tuple(np.where(Y[i]==0))
	Y[i][training_with_two] = -1
XTest = testset[:,0:5]
yTest = testset[:,5]
testing_with_two = tuple(np.where(yTest==0))
yTest[testing_with_two] = -1


#K-fold for selection of hyperparameter C in SVM
accuracy_set = []
C = []
accuracy_mean = []
accuracy_var = []
for c in range(2, 80):
	for i in range(k):
		X_train = np.concatenate([x for xi,x in enumerate(X) if xi!= i])
		y_train = np.concatenate([y for yi,y in enumerate(Y) if yi!= i])
		clf = svm.SVC(C=c/20.0,gamma='scale')
		clf.fit(X_train, y_train)
		(n, dx) = X[i].shape
		correct = 0.0
		for j in range(n):
			pred = clf.predict(X[i][j].reshape((1,5)))
			if Y[i][j] == pred:
				correct += 1.0
		accuracy = correct/n
		accuracy_set.append(accuracy)
	accuracy_mean.append(np.mean(accuracy_set))
	accuracy_var.append(np.var(accuracy_set))
	C.append(c/20.0)
#print "C =", c/20.0,"mean", accuracy_mean, "var", accuracy_var
print "done"
#plot (SVM)
pp.figure()
pp.plot(C, accuracy_var, color = "red")
pp.xlabel('C')
pp.ylabel('Var')
pp.savefig('/homes/zhao684/cs373/Var (SVM)', dpi=150)
pp.figure()
pp.plot(C, accuracy_mean, color = "green")
pp.xlabel('C')
pp.ylabel('Mean')
#CHANGE THE PWD when run on your computer!!!
pp.savefig('/homes/zhao684/cs373/Mean (SVM)', dpi=150)


#K-fold for selection of 
accuracy_set = []
WClassifier = []
accuracy_mean = []
accuracy_var = []

for w in range(5,100,5):
	for i in range(k):
		X_train = np.concatenate([x for xi,x in enumerate(X) if xi!= i])
		y_train = np.concatenate([y for yi,y in enumerate(Y) if yi!= i])
		accuracy_set.append(inter(X_train,y_train,w))
	accuracy_mean.append(np.mean(accuracy_set))
	accuracy_var.append(np.var(accuracy_set))
	print "Mean: ", np.mean(accuracy_set), 'Var' , np.var(accuracy_set)
	WClassifier.append(w)
print "done"
#plot accuracy (AdaBoost)
pp.figure()
pp.plot(WClassifier, accuracy_var, color = "red")
pp.xlabel('# of Weak Classifier')
pp.ylabel('Var')
pp.savefig('/homes/zhao684/cs373/Var (AdaBoost)', dpi=150)
pp.figure()
pp.plot(WClassifier, accuracy_mean, color = "green")
pp.xlabel('# of Weak Classifier')
pp.ylabel('Mean')
pp.savefig('/homes/zhao684/cs373/Mean (AdaBoost)', dpi=150)
