from typing import Union
import pandas as pd
import numpy as np

def accuracy(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the accuracy
    """
    assert y_hat.size == y.size, "Size of predictions and true values must be equal"
    
    correct_predictions = (y_hat == y).sum()
    total_predictions = y.size
    
    return correct_predictions / total_predictions

def precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the precision
    """
    assert y_hat.size == y.size, "Size of predictions and true values must be equal"
    
    true_positive = ((y_hat == cls) & (y == cls)).sum()
    false_positive = ((y_hat == cls) & (y != cls)).sum()
    
    if (true_positive + false_positive) == 0:
        return 0.0
    
    return true_positive / (true_positive + false_positive)

def recall(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the recall
    """
    assert y_hat.size == y.size, "Size of predictions and true values must be equal"
    
    true_positive = ((y_hat == cls) & (y == cls)).sum()
    false_negative = ((y_hat != cls) & (y == cls)).sum()
    
    if (true_positive + false_negative) == 0:
        return 0.0
    
    return true_positive / (true_positive + false_negative)

def rmse(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the root-mean-squared-error (RMSE)
    """
    assert y_hat.size == y.size, "Size of predictions and true values must be equal"
    
    mse = ((y_hat - y) ** 2).mean()
    return np.sqrt(mse)

def mae(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the mean-absolute-error (MAE)
    """
    assert y_hat.size == y.size, "Size of predictions and true values must be equal"
    
    return (y_hat - y).abs().mean()
