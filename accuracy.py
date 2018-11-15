import numpy as np
import matplotlib as plt
#set backend for matplotlib
plt.use('Agg')
import matplotlib.pyplot as pp
#install sklearn first : using command "pip install -U scikit-learn"
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_gaussian_quantiles


def getSVMAccuracy(X,y):
	clf = svm.SVC(gamma='scale')
	clf.fit(X, y) 
	(n, dx) = XTest.shape
	correct = 0.0
	for i in range(n):
		pred = clf.predict(XTest[i].reshape((1,5)))
		if yTest[i] == clf.predict(XTest[i].reshape((1,5))):
			correct += 1.0
	accuracy = correct/n
	return accuracy

def getBoostAccuracy(X,y):
	bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),algorithm="SAMME",n_estimators=5)
	bdt.fit(X, y)
	(n, dx) = XTest.shape
	correct = 0.0
	for i in range(n):
		if yTest[i] == bdt.predict(XTest[i].reshape((1,5))):
			correct += 1.0
	accuracy = correct/n
	return accuracy
	
	
# reading the original format of the data
Input = np.loadtxt('datatraining.txt', delimiter=',', skiprows=1, usecols=range(2,8))
Input.reshape((-1,6))
# shuffle training set 
np.random.shuffle(Input)
X=Input[:,0:5].reshape((-1,5))
y=Input[:,5].reshape((-1,1))


# reading the testdata (using only datatest.txt for now)
XTest = np.loadtxt('datatest.txt', delimiter=',', skiprows=1, usecols=range(2,7))
yTest = np.loadtxt('datatest.txt', delimiter=',', skiprows=1, usecols=range(7,8))
XTest = XTest.reshape((-1,5))
yTest = yTest.reshape((-1,1))

# convert 0 to -1 for both training and testing dataset
training_with_two = list(np.where(y==0)[0])
y[training_with_two] = -1

testing_with_two = list(np.where(yTest==0)[0])
yTest[testing_with_two] = -1

p_sample = list(np.where(y==1)[0])
n_sample = list(np.where(y==-1)[0])

#print X,y

# change matrix to array
y = np.squeeze(np.asarray(y))

#get accuracies of different training data size with the same testing dataset
n,d = X.shape
'''
accuracy = []
index = []

#for i in range(700,1000,10): # graph for training size 700~1000, step 10
for i in range(100,n,200):	#graph for 100 to the size of training dataset (about 8000), step 200
	index.append(i)
	accuracy.append(getSVMAccuracy(X[0:i,:],y[0:i]))
accuracy = np.array(accuracy)
index = np.array(index)

#plot accuracy (SVM)
pp.figure()
pp.plot(index, accuracy, 'go')
pp.xlabel('training dataset size')
pp.ylabel('accuracy')
#CHANGE THE PWD when run on your computer!!!
pp.savefig('/homes/zhao684/cs373/project', dpi=150)
'''

accuracy = []
index = []
#for i in range(700,1000,10): # graph for training size 700~1000, step 10
for i in range(100,n,200):	#graph for 100 to the size of training dataset (about 8000), step 200
	index.append(i)
	accuracy.append(getBoostAccuracy(X[0:i,:],y[0:i]))
accuracy = np.array(accuracy)
index = np.array(index)

print accuracy
#plot accuracy (AdaBoost)
pp.figure()
pp.plot(index, accuracy, 'go')
pp.xlabel('training dataset size')
pp.ylabel('accuracy')
#CHANGE THE PWD when run on your computer!!!
pp.savefig('/homes/zhao684/cs373/project', dpi=150)
