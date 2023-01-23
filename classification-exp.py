import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from sklearn.datasets import make_classification

np.random.seed(42)
X, y = make_classification(n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5)

plt.scatter(X[:, 0], X[:, 1], c=y)

X_train,y_train = X[:int(0.7*len(X))],y[:int(0.7*len(y))]
X_test,y_test = X[int(0.7*len(X)):],y[int(0.7*len(y)):]

for criteria in ['information_gain','gini_index']:
    print("criteria: ", criteria)
    tree = DecisionTree(criterion = criteria,max_depth=4)
    tree.fit(pd.DataFrame(X_train),pd.Series(y_train))
    y_predict = tree.predict(pd.DataFrame(X_test))
    print('\nDecision Tree\n')
    tree.plot()
    acc = accuracy(pd.Series(y_predict),pd.Series(y_test))
    print("Accuracy: ",acc)
    for cls in pd.Series(y).unique():
        print("class: ", cls)
        p = precision(pd.Series(y_predict), pd.Series(y_test), cls)
        r = recall(pd.Series(y_predict), pd.Series(y_test), cls)

        print("Precision: ", p)
        print("Recall: ", r)

print("\nUsing 5-fold validation\n")

for criteria in ['information_gain','gini_index']:
    print('criteria:',criteria)
    y_predict = []
    for i in range(5):
        left = int(len(X)*(i/5))
        right = int(len(X)*(i+1)/5)
        X_train = np.concatenate((X[:left],X[right:]),axis=0)
        X_test = X[left:right]
        y_train = np.concatenate((y[:left],y[right:]),axis = 0)
        y_test = y[left:right]
        tree= DecisionTree(criterion=criteria, max_depth = 4)
        tree.fit(pd.DataFrame(X_train),pd.Series(y_train))
        y_part = tree.predict(pd.DataFrame(X_test))
        y_predict += y_part.tolist()
    acc = accuracy(pd.Series(y_predict),pd.Series(y))
    print("Accuracy: ",acc)
    for cls in pd.Series(y).unique():
        print("class: ", cls)
        p = precision(pd.Series(y_predict), pd.Series(y), cls)
        r = recall(pd.Series(y_predict), pd.Series(y), cls)

        print("Precision: ", p)
        print("Recall: ", r)

print("\nNested Cross Validation\n")

X = X[:int(len(X)*0.8)]
y = y[:int(len(y)*0.8)]
optimum_depth = 0
max_acc = 0
for j in range(tree.max_depth+1):
    y_predict = []
    for i in range(5):
        left = int(len(X)*(i/5))
        right = int(len(X)*(i+1)/5)
        X_train = np.concatenate((X[:left],X[right:]),axis=0)
        X_test = X[left:right]
        y_train = np.concatenate((y[:left],y[right:]),axis =0)
        y_test = y[left:right]
        tree= DecisionTree(criterion = 'gini_index', max_depth = j)
        tree.fit(pd.DataFrame(X_train),pd.Series(y_train))
        y_part = tree.predict(pd.DataFrame(X_test))
        y_predict += y_part.tolist()
    acc = accuracy(pd.Series(y_predict),pd.Series(y))
    if max_acc<acc:
        max_acc = acc
        optimum_depth = j

print("Optimum Depth: ", optimum_depth)
print("Accuracy at optimal depth: ",max_acc)
