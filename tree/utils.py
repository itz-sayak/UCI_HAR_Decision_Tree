import pandas as pd
import numpy as np
from typing import Tuple, Union, Any

def one_hot_encoding(X: pd.DataFrame) -> pd.DataFrame:
    """
    Function to perform one hot encoding on the input data
    """
    return pd.get_dummies(X)

def check_if_real(y: pd.Series) -> bool:
    """
    Function to check if the given series has real or discrete values
    """
    return pd.api.types.is_numeric_dtype(y) and not pd.api.types.is_integer_dtype(y)


def entropy(Y: pd.Series) -> float:
    """
    Function to calculate the entropy
    """
    proportions = Y.value_counts(normalize=True)
    return -np.sum(proportions * np.log2(proportions + 1e-9))

def gini_index(Y: pd.Series) -> float:
    """
    Function to calculate the Gini index
    """
    proportions = Y.value_counts(normalize=True)
    return 1 - np.sum(proportions ** 2)

def mean_squared_error(Y: pd.Series) -> float:
    """
    Function to calculate the mean squared error
    """
    return np.var(Y)

def information_gain(Y: pd.Series, attr: pd.Series, criterion: str) -> float:
    """
    Function to calculate the information gain using criterion (entropy, gini index or MSE)
    """
    if criterion in ["entropy", "gini_index"]:
        overall_entropy = entropy(Y) if criterion == "entropy" else gini_index(Y)
        unique_values = attr.unique()
        weighted_entropy = 0
        for value in unique_values:
            subset_Y = Y[attr == value]
            weight = len(subset_Y) / len(Y)
            weighted_entropy += weight * (entropy(subset_Y) if criterion == "entropy" else gini_index(subset_Y))
        return overall_entropy - weighted_entropy
    elif criterion == "mse":
        overall_mse = mean_squared_error(Y)
        unique_values = attr.unique()
        weighted_mse = 0
        for value in unique_values:
            subset_Y = Y[attr == value]
            weight = len(subset_Y) / len(Y)
            weighted_mse += weight * mean_squared_error(subset_Y)
        return overall_mse - weighted_mse

def opt_split_attribute(X: pd.DataFrame, y: pd.Series, criterion: str, features: pd.Series) -> Tuple[Union[str, None], float]:
    """
    Function to find the optimal attribute to split upon.
    """
    best_feature = None
    best_threshold = None
    best_score = -float('inf') if criterion in ["entropy", "gini_index"] else float('inf')
    
    for feature in features:
        if check_if_real(X[feature]):
            thresholds = X[feature].unique()
            for threshold in thresholds:
                left_y, right_y = split_data(X, y, feature, threshold)
                score = information_gain(y, X[feature], criterion) if criterion in ["entropy", "gini_index"] else mean_squared_error(y)
                if (criterion in ["entropy", "gini_index"] and score > best_score) or (criterion == "mse" and score < best_score):
                    best_feature = feature
                    best_threshold = threshold
                    best_score = score
        else:
            score = information_gain(y, X[feature], criterion) if criterion in ["entropy", "gini_index"] else mean_squared_error(y)
            if (criterion in ["entropy", "gini_index"] and score > best_score) or (criterion == "mse" and score < best_score):
                best_feature = feature
                best_threshold = None
                best_score = score

    return best_feature, best_threshold


def split_data(X: pd.DataFrame, y: pd.Series, attribute: str, value: Any) -> Tuple[pd.Series, pd.Series]:
    """
    Function to split the data according to an attribute.
    """
    if pd.api.types.is_numeric_dtype(X[attribute]):
        # Numeric split
        left_mask = X[attribute] <= value
        right_mask = X[attribute] > value
    else:
        # Categorical split
        left_mask = X[attribute] == value
        right_mask = X[attribute] != value
    
    return y[left_mask], y[right_mask]

