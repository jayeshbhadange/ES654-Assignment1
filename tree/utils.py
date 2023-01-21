import pandas as pd
import math

def entropy(Y: pd.Series) -> float:
    """
    Function to calculate the entropy
    """
    Y=list(Y)
    label_counts = {}
    length=len(Y)
    for i in Y:
      if i in label_counts:
        label_counts[i]+=1
      else:
        label_counts[i]=1

    entropy=0
    for val in label_counts.values():
      prob=(val/length)
      prob=prob*math.log2(prob)
      entropy-=prob
    return float(entropy)  





def gini_index(Y: pd.Series) -> float:
    """
    Function to calculate the gini index
    """
    Y=list(Y)
    label_counts = {}
    length=len(Y)
    for i in Y:
      if i in label_counts:
        label_counts[i]+=1
      else:
        label_counts[i]=1

    gini=1
    for val in label_counts.values():
      prob=val/length
      gini-=prob**2
    return float(gini)  
    


def information_gain(Y: pd.Series, attr: pd.Series) -> float:
    """
    Function to calculate the information gain
    """
    ent_Y=entropy(Y)
    Y=list(Y)
    length=len(Y)
    att=list(attr)
    lab={}
    for i in att:
      if i in lab:
        lab[i].append(Y[i])
      else:
        lab[i]=[Y[i]]

    for val in lab.values():
      ent_Y-=(len(val)/length)*entropy(pd.Series(val))
    return ent_Y  
