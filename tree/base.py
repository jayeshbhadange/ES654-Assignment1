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
from utils import entropy, information_gain, gini_index

np.random.seed(42)

class Node():
    def __init__(self):
        self.typeofatt=None         #dtrue if discrete attributeHere
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
      # classification if y is discrete
      if(y.dtype=="category"):
        clasNo=np.unique(y)
        if(clasNo.size==1): # only one value for prediction
          node.isleaf = True
          node.isAttrCategory = True # for discrete attribute
          node.value = clasNo[0]
          return node
        if(self.max_depth==depth): #if max depth is reached
          node.isleaf = True
          node.isAttrCategory = True
          node.value = np.bincount(y).argmax()
          return node


        
        for i in x:
          att=x[i]
          if(att.dtype=="category"): # checking for deicrete attributes
            if(self.criterion=="information_gain"):
              y1=pd.Series(y)
              gain=information_gain(y1,att)
              
            else:
              gain=0
              
              y1=list(y)
              att=list(att)
              length=len(att)
              lab={}
              for i in att:
                if i in lab:
                  lab[i].append(y1[i])
                else:
                  lab[i]=[y1[i]]

              for val in lab.values():
                gain-=(len(val)/length)*gini_index(pd.Series(val))
            if(gain>maxEnt):
                maxEnt=gain
                best_attribute=i
          else: # if real attributes
            att=att.sort_values(ascending=True)
            for j in range(att.shape[0]-1):
              split=(att[j]+att[j+1])/2
              left= pd.Series([y[k] for k in range(y.size) if att[k]<=split])
              right = pd.Series([y[k] for k in range(y.size) if att[k]>split])
              if(self.criterion=="information_gain"):
                y1=pd.Series(y)
                initial_entropy = entropy(y1)
                
                
                left_entropy = entropy(left)
                right_entropy = entropy(right)
                gian= initial_entropy - (left_entropy * (left.size / len(y)) + right_entropy * (right.size / len(y)))
              else:
                gain = (-1/y.size)*(left.size*gini_index(left) + right.size*gini_index(right))
              if(gain>maxEnt):
                maxEnt=gain
                best_attribute=i
                splitval=split

      else: #means regression
        if(self.max_depth==depth):
          node.isleaf=True
          node.value=y.mean()
          return node

        for i in x:
          att=x[i]
          if(att.dtype=="category"): # checking for discrete attributes
              att=list(att)
              label = {}
              length=len(y)
              for i in att:
                if i in label.keys():
                  label[i].append(y[i])
                else:
                  label[i]=[y[i]]
              gain=0    
              for val in lab.values():
                gain+=len(val)*((pd.series(val)).var())
              tot_var=np.var(y)
              tot_var-=gain/len(y)
              if(tot_var>maxEnt):
                maxEnt=tot_var
                best_attribute=i
          else: # real input
            att=att.sort_values(ascending=True)
            for j in range(att.shape[0]-1):
              split=(att[j]+att[j+1])/2
              left= ([y[k] for k in range(y.size) if att[k]<=split])
              left=np.asarray(left)

              right = ([y[k] for k in range(y.size) if att[k]>split])
              right=np.asarray(right)
              err=np.sum(np.square(np.mean(left)-left)) + np.sum(np.square(np.mean(right)-right))
              if(err<maxEntReal):
                maxEntReal=err
                best_attribute=i
                splitval=split
      if(splitval==None):
        node.typeofatt=True
        node.att=best_attribute
        for i in x[best_attribute].unique():
          val=[]
          for j in range(len(x[best_attribute])):
            if x[best_attribute][j]==i:
              val.append(y[j])
          x_modify=x[x[best_attribute]==i].reset_index().drop(['index',best_attribute],axis=1) # modifying x 
          val=pd.Series(val,dtype=y.dtype)
          node.child[i]=self.fit_tree(x_modify, val, depth+1)

      else:
        node.att=best_attribute
        node.split_val=splitval
        val_left=[]
        val_right=[]
        x_left = x[x[best_attribute]<=splitval].reset_index().drop(['index'],axis=1)
        x_right = x[x[best_attribute]>splitval].reset_index().drop(['index'],axis=1)
        for j in range(len(x[best_attribute])):
          if x[best_attribute][j]<=splitval:
            val_left.append(y[j])

          else:
            val_right.append(y[j])
        val_left=pd.Series(val_left,dtype=y.dtype)  
        val_right=pd.Series(val_right,dtype=y.dtype)   
        node.child["left"]=self.fit_tree(x_left, val_left, depth+1)
        node.child["right"]=self.fit_tree(x_right, val_right, depth+1)
      return node

          



    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Function to train and construct the decision tree
        """
        y=y.to_numpy()
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
        
