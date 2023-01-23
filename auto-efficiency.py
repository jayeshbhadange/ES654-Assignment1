import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
from sklearn import preprocessing 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

np.random.seed(42)

# Read dataset
data = pd.read_fwf('auto-mpg.data',names = ["mpg", "cylinders", "displacement", "horsepower", "weight", "accelaration", "model year", "origin", "car name"])
# we don't need the string columns in our model
data.drop('car name',axis = 1,inplace = True)
# removing the unknown data points
data.drop(data.index[data['horsepower']=='?'],inplace = True)

for i in data.columns:
    data[i] = data[i].astype('float64')
data.reset_index(drop = True,inplace = True)

X_train, X_test, y_train, y_test = data.iloc[:n,1:], data.iloc[n+1:,1:], data.iloc[:n,0], data.iloc[n+1:,0]
print("\nOur implementation of Decision Tree\n")
for criteria in ['information_gain','gini_index']:
    print("criteria: ", criteria)

    i_tree = DecisionTree(criterion = criteria,max_depth=4)
    i_tree.fit(pd.DataFrame(X_train),pd.Series(y_train))
    y_predict = i_tree.predict(pd.DataFrame(X_test))
    i_tree.plot()
    acc = accuracy(pd.Series(y_predict),pd.Series(y_test))
    print("Accuracy: ",acc)

    print("RMSE: ",rmse(y_predict,y_test))
    print("MAE: ",mae(y_predict,y_test))
print ('\n')
sk_tree = DecisionTreeRegressor(max_depth=7)
sk_tree.fit(X_train, y_train)
Tree_text = tree.export_text(sk_tree)
print(Tree_text)
y_sk_predict = sk_tree.predict(X_test)
y_sk_predict = pd.Series(y_sk_predict)
print('RMSE between our model and Decision tree module from sklearn : ', rmse(y_sk_predict, y_test))
print('MAE between our model and Decision tree module from sklearn : ', mae(y_sk_predict, y_test))
