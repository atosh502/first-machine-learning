#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 06:29:22 2017

@author: atosh
"""

import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names = names)

# dimension of the data
# print the no of data - rows and no of attributes - cols
print(dataset.shape)

# a peek into the data
# display the first 20 data
print(dataset.head(200))

# description about the data
print(dataset.describe())

# distribute according to the class
print(dataset.groupby('class').size())

# box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()

# histograms
dataset.hist()
plt.show()

# scatter plot matrix
scatter_matrix(dataset)
plt.show();

# creating a validation dataset
# seperate the dataset such that 80% is used to train
# and the rest 20% will be used as a validation dataset
array = dataset.values

# assign only the size of sepal and petal to X
X = array[:, 0:4]

# assign only the type of iris to Y
Y = array[:, 4]

validation_size = 0.20
seed = 7
        
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size = validation_size, random_state = seed)

# test harness
# test options and evaluation metric
# it is the ratio of correctly predicted instances divided by the total
# number of instances in the dataset mulitplied by 100

seed = 7
scoring = 'accuracy'

# spot check algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

# evaluating each model in turn
results = []
names = []
for name, model in models:
    # split the dataset into k consecutive folds and each fold is used once as
    # a validation and k-1 folds are used for training
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    
    # evaluate a score by cross validation
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" %(name, cv_results.mean(), cv_results.std())
    print(msg)

# now compare the algorithms in a box and whisker plot

fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

# let's make some predictions using the KNN algorithm
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))


