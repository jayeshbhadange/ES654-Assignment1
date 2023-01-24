
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
#from tree.base import DecisionTree
#from metrics import *

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
  N = np.random.randint(1,150,size=3)
  M = np.random.randint(1,50,size=3)

  d={}
  p={}
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
        if m in d.keys():
          d[m].append((n,end-start))
        else:
          d[m]=[(n,end-start)]  
        if m in p.keys():
          p[m].append((n,en2-st2))
        else:
          p[m]=[(n,en2-st2)]        


  fig,ax=plt.subplots(1,len(d))  
  j=0
  for i in d.keys():
    x,y=zip(*d[i]) 
    ax[j].plot(x,y)
    ax[j].set_xlabel(f"N at attributes={i}")
    ax[j].set_ylabel("time")
    j+=1
  plt.subplots_adjust(wspace=4,hspace=4)
  fig.suptitle(f"{category}, fit time")
  plt.show()    
  fig,ax=plt.subplots(1,len(p))  
  j=0
  for i in p.keys():
    x,y=zip(*p[i]) 
    ax[j].plot(x,y)
    ax[j].set_xlabel(f"N at attributes={i}")
    ax[j].set_ylabel("time")    
    j+=1
  plt.subplots_adjust(wspace=4,hspace=4)
  fig.suptitle(f"{category}, prediction time")
  plt.show()      
  
#caltime("DisDis")
#caltime("ReDis")
caltime("DisRe")
#caltime("ReRe")
