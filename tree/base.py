"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tree.utils import entropy, information_gain, gini_index

np.random.seed(42)

class Node():
    def __init__(self):
        self.typeofatt=False        #true if discrete attributeHere
        self.split_val=None         #Spliting number for Real input
        self.value=None             #Output value
        self.child={}               #child nodes           
        self.isleaf=False
        self.att=None
@dataclass
class DecisionTree:
    criterion: Literal["information_gain", "gini_index"]  # criterion won't be used for regression
    max_depth: int  # The maximum depth the tree can grow to
    root=None
    def fit_tree(self,x,y,depth):
      node=Node()
      maxEnt = -1*float("inf")
      maxEntReal=float("inf")
      best_attribute = -1
      splitval=None
      y1=y.to_numpy()
      # classification if y is discrete
      if(y.dtype=="category"):

        clasNo=np.unique(y)
        if(clasNo.size==1): # only one value for prediction
          node.isleaf = True #predict directly
          node.typeofatt = True # for discrete attribute
          node.value = np.random.choice(clasNo)
          return node
        if(self.max_depth==depth or x.shape[1]==0): #if max depth is reached
          node.isleaf = True
          node.typeofatt = True
          node.value = np.bincount(y1).argmax()
          return node
        
        for i in x:
          att=x[i]
          if(att.dtype=="category"): # checking for deicrete attributes
            if(self.criterion=="information_gain"):
              
              gain=information_gain(y,att)
              
            else:
              gain=0
              
              
              att=list(att)
              length=len(att)
              lab={}
              for k in range(len(att)):
                if att[k] in lab.keys():
                  lab[att[k]].append(y1[k])
                else:
                  lab[att[k]]=[y1[k]]

              for val in lab.values():
                gain-=(len(val)/length)*gini_index(pd.Series(val))
            if(gain>maxEnt):
                maxEnt=gain
                best_attribute=i
          else: # if real attributes
            att=att.sort_values(ascending=True)
            for j in range(att.shape[0]-1):
              gain=None
              split=(att[j]+att[j+1])/2
              left= pd.Series([y1[k] for k in range(y1.size) if att[k]<=split])
              right = pd.Series([y1[k] for k in range(y1.size) if att[k]>split])
              if(self.criterion=="information_gain"):
                
                initial_entropy = entropy(y)
                
                
                left_entropy = entropy(left)
                right_entropy = entropy(right)
                gain= initial_entropy - (left_entropy * (left.size / len(y1)) + right_entropy * (right.size / len(y1)))
              else:
                gain = (-1/len(y1))*((left.size*gini_index(left) + right.size*gini_index(right)))
              if(gain!=None):
                if(gain>maxEnt):
                  maxEnt=gain
                  best_attribute=i
                  splitval=split
              else:
                  gain=maxEnt+10000
                  best_attribute=i
                  splitval=split
      else: #means regression
        if(self.max_depth==depth or y1.size==1 or x.shape[1]==0):
          node.isleaf=True
          node.value=y1.mean()
          return node
        
        for i in x:
          att=x[i]
          if(att.dtype=="category"): # checking for discrete attributes
              uniqval = np.unique(att)
              gain=0
              for j in uniqval:
                  y_sub = pd.Series([y[k] for k in range(y.size) if att[k]==j])
                  gain += y_sub.size*np.var(y_sub)
                  if(maxEntReal>gain):
                            maxEntReal = gain
                            best_attribute = i
                            splitval = None

          else: # real input
            att=att.sort_values(ascending=True)
            for j in range(y.shape[0]-1):
              split=(att[j]+att[j+1])/2
              left=[]
              right=[]
              
              left= ([y[k] for k in range(y.size) if att[k]<=split])
              left=np.asarray(left)
              
              right = ([y[k] for k in range(y.size) if att[k]>split])
              right=np.asarray(right)
              errr=np.sum(np.square(np.mean(left)-left)) + np.sum(np.square(np.mean(right)-right))
              if(errr<maxEntReal):
                maxEntReal=errr
                best_attribute=i
                splitval=split
      if(splitval==None):
        node.typeofatt=True
        node.att=best_attribute
        
        classes = np.unique(x[best_attribute])
      
        for j in classes:
          y_modify = pd.Series([y1[k] for k in range(y1.size) if x[best_attribute][k]==j], dtype=y1.dtype)
          x_modify=x[x[best_attribute]==j].reset_index().drop(['index',best_attribute],axis=1) 
          node.child[j]=self.fit_tree(x_modify, y_modify, depth+1)

        
      else:
        node.att=best_attribute
        node.split_val=splitval
        val_left=[]
        val_right=[]
        x_left = x[x[best_attribute]<=splitval].reset_index().drop(['index'],axis=1)
        x_right = x[x[best_attribute]>splitval].reset_index().drop(['index'],axis=1)
        for j in range(len(x[best_attribute])):
          if x[best_attribute][j]<=splitval:
            val_left.append(y1[j])

          else:
            val_right.append(y1[j])
        val_left=pd.Series(val_left)  
        
        val_right=pd.Series(val_right)
          
        node.child["left"]=self.fit_tree(x_left, val_left, depth+1)
        node.child["right"]=self.fit_tree(x_right, val_right, depth+1)
      return node

          



    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Function to train and construct the decision tree
        """
        #y=y.to_numpy()
        self.root=self.fit_tree(X,y,0)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Funtion to run the decision tree on test inputs
        """
        output=list()
        for i in range(X.shape[0]):
          x = X.iloc[i,:] #selecting ith row for prediction
          head=self.root
          while(not head.isleaf):
            if(head.typeofatt==True) :# discrete attribute
              head=head.child[x[head.att]]
            else:                     #real att
              if(x[head.att]<=head.split_val):
                head = head.child["left"]
              else:
                head = head.child["right"]  
          output.append(head.value)
        return pd.Series(output)  



    def plot_tree(self,head,depth):
      if head.isleaf:
          print("prediction ",head.value)
      else:
        if(head.typeofatt) : # if attribute is discrete
          for i in head.child.keys() :
            print(f"?{head.att} == {i}")
            print("\t"*(depth+1),end="")
            self.plot_tree(head.child[i],depth+1)
            print("\t"*depth,end="")
        else: #real attribte
          print(f"?(X[{head.att}] > {head.split_val})")
          print("\t"*(depth+1),"Y:= ",end="")
          self.plot_tree(head.child["left"],depth+1)
          print("\t"*(depth+1),"N:= ",end="")  
          self.plot_tree(head.child["right"],depth+1)    


    def plot(self) -> None:
        """
        Function to plot the tree

        Output Example:
        ?(X1 > 4)
            Y: ?(X2 > 7)
                Y: Class A
                N: Class B
            N: Class C
        Where Y => Yes and N => No
        """
        head=self.root
        self.plot_tree(head,0)
