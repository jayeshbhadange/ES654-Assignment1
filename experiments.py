
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *

np.random.seed(42)
num_average_time = 100

# Learn DTs 
# ...
# 
# Function to calculate average time (and std) taken by fit() and predict() for different N and P for 4 different cases of DTs
# ...
# Function to plot the results
# ..
# Function to create fake data (take inspiration from usage.py)
# ...
# ..other functions

def learndata(N,M,category): # create fake dataset for four cases
  if(category=="DisDis"):
    X = pd.DataFrame({i:pd.Series(np.random.randint(M, size = N), dtype="category") for i in range(5)})
    y = pd.Series(np.random.randint(M, size = N),  dtype="category")
  elif(category=="ReDis"):
    X = pd.DataFrame(np.random.randn(N, M))
    y = pd.Series(np.random.randint(M, size = N), dtype="category")
  elif(category=="ReRe"):
    X = pd.DataFrame(np.random.randn(N, M))
    y = pd.Series(np.random.randn(N))
  else:
    X = pd.DataFrame({i:pd.Series(np.random.randint(M, size = N), dtype="category") for i in range(5)})
    y = pd.Series(np.random.randn(N))
  return X,y

def caltime(category):
  np.random.seed(42)
  N = np.random.randint(1,150,size=8)
  M = np.random.randint(1,50,size=5)
  fittime=[]
  predtime=[]
  for n in N:
    for m in M:
      X,y=learndata(n,m,category)
      for criteria in ['information_gain', 'gini_index']:
        Dtree = DecisionTree(criterion=criteria,max_depth=4) #Split based on Inf. Gain
        start=time.time()
        Dtree.fit(X, y)
        end=time.time()
        st2=time.time()
        Dtree.predict(X)
        en2=time.time()
        fittime.append((n,m,end-start))
        predtime.append((n,m,en2-st2))
  fig, axs = plt.subplots(1, 2, figsize=(10, 5))
  axs[0].scatter(*zip(*fittime))
  axs[0].set_title("Fit time")
  axs[1].scatter(*zip(*predtime))
  axs[1].set_title("Predict time")
  plt.show()
caltime("DisDis")
