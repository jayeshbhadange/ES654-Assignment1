from typing import Union
import pandas as pd
import numpy as np


def accuracy(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the accuracy
    """

    """
    The following assert checks if sizes of y_hat and y are equal.
    Students are required to add appropriate assert checks at places to
    ensure that the function does not fail in corner cases.
    """
    assert y_hat.size == y.size
    y_hat=y_hat.to_numpy()
    y=y.to_numpy()
    val=y_hat-y
    cnt=np.count_nonzero(val== 0)
    return float(cnt/len(y))


def precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the precision
    """
    assert y_hat.size == y.size
    y_hat=y_hat.to_numpy()
    y=y.to_numpy()
    tp = np.sum((y == cls) & (y_hat == cls))
    fp = np.sum((y != cls) & (y_hat == cls))
    return float(tp / (tp + fp))


def recall(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the recall
    """
    assert y_hat.size == y.size
    y_hat=y_hat.to_numpy()
    y=y.to_numpy()
    tp = np.sum((y == cls) & (y_hat == cls))
    fn = np.sum((y == cls) & (y_hat != cls))
    return float(tp / (tp + fn))
    


def rmse(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the root-mean-squared-error(rmse)
    """
    assert y_hat.size == y.size
    y_hat=y_hat.to_numpy()
    y=y.to_numpy()
    err=(np.sqrt(np.mean((y_hat-y)**2)))
    return float(err)


def mae(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the mean-absolute-error(mae)
    """
    assert y_hat.size == y.size
    y_hat=y_hat.to_numpy()
    y=y.to_numpy()
    err=(np.mean(abs(y_hat-y)))
    return err

    
