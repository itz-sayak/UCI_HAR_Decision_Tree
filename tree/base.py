"""
DecisionTree Class Implementation

This module contains the implementation of a Decision Tree classifier and regressor. The `DecisionTree` class can be 
used for both classification and regression tasks, depending on the criterion specified. It supports the following criteria:
- "information_gain" for classification using entropy.
- "gini_index" for classification using the Gini index.
- "mse" for regression using the mean squared error.

Classes:
    DecisionTree: A class to build and train a decision tree.

Functions:
    fit(X, y): Trains the decision tree on the provided dataset.
    predict(X): Predicts the class labels or values for the provided data.
    plot(): Plots the structure of the decision tree.
    get_params(deep=True): Returns the parameters of the DecisionTree object.
    set_params(**params): Sets the parameters of the DecisionTree object.
"""

import numpy as np
import pandas as pd
from typing import Literal, Union, Tuple, Any
from dataclasses import dataclass
from .utils import entropy, gini_index, mean_squared_error

@dataclass
class DecisionTree:
    """
    DecisionTree(criterion: Literal["information_gain", "gini_index"], max_depth: int = 5)

    A class to build and train a decision tree classifier or regressor.

    Attributes:
        criterion (str): The function to measure the quality of a split. Supported criteria are "information_gain", "gini_index", and "mse".
        max_depth (int): The maximum depth of the tree. Default is 5.
        tree_ (Any): The structure of the trained decision tree.

    Methods:
        fit(X, y): Trains the decision tree on the provided dataset.
        _fit(X, y, depth): Recursive function to build the tree.
        _best_split(X, y): Finds the best feature and threshold to split the data.
        split_data(X, y, attribute, value): Splits the data according to an attribute.
        _calculate_score(left_y, right_y): Calculates the score of a split.
        _information_gain(left_y, right_y): Computes the information gain of a split.
        _gini_index(left_y, right_y): Computes the Gini index of a split.
        _mean_squared_error(left_y, right_y): Computes the mean squared error (MSE) of a split.
        _most_common_label(y): Returns the most common label in the series.
        predict(X): Predicts the class labels or values for the provided data.
        _predict(sample, tree): Traverses the tree to get the prediction for a single sample.
        plot(): Plots the structure of the decision tree.
        get_params(deep=True): Returns the parameters of the DecisionTree object.
        set_params(**params): Sets the parameters of the DecisionTree object.
    """

    def __init__(self, criterion: Literal["information_gain", "gini_index"], max_depth: int = 5):
        """
        Initializes the DecisionTree with a given criterion and maximum depth.

        Args:
            criterion (str): The function to measure the quality of a split.
            max_depth (int): The maximum depth of the tree.
        """
        self.criterion = criterion
        self.max_depth = max_depth
        self.tree_ = None

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> None:
        """
        Trains the decision tree on the provided dataset.

        Args:
            X (Union[np.ndarray, pd.DataFrame]): Feature matrix.
            y (Union[np.ndarray, pd.Series]): Target vector.
        """
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        if isinstance(y, np.ndarray):
            y = pd.Series(y)
        
        self.X_ = X
        self.y_ = y
        self.n_features_ = X.shape[1]
        self.tree_ = self._fit(X, y, depth=0)

    def _fit(self, X: pd.DataFrame, y: pd.Series, depth: int) -> Any:
        """
        Recursive function to build the decision tree.

        Args:
            X (pd.DataFrame): Feature matrix.
            y (pd.Series): Target vector.
            depth (int): Current depth of the tree.

        Returns:
            Any: The structure of the tree node or the label/value if a leaf node is reached.
        """
        unique_classes = y.unique()
        if len(unique_classes) == 1:
            return unique_classes[0]
        
        if depth >= self.max_depth:
            return self._most_common_label(y)
        
        if len(X) == 0:
            return self._most_common_label(y)
        
        best_feature, best_threshold, best_score = self._best_split(X, y)
        
        if best_feature is None:
            return self._most_common_label(y)
        
        if pd.api.types.is_numeric_dtype(X[best_feature]):
            left_indices = X[best_feature] <= best_threshold
            right_indices = X[best_feature] > best_threshold
        else:
            left_indices = X[best_feature] == best_threshold
            right_indices = X[best_feature] != best_threshold
        
        if left_indices.sum() == 0 or right_indices.sum() == 0:
            return self._most_common_label(y)

        left_tree = self._fit(X[left_indices], y[left_indices], depth + 1)
        right_tree = self._fit(X[right_indices], y[right_indices], depth + 1)
        
        return (best_feature, best_threshold, left_tree, right_tree)

    def _best_split(self, X: pd.DataFrame, y: pd.Series) -> Tuple[str, Any, float]:
        """
        Finds the best feature and threshold to split the data.

        Args:
            X (pd.DataFrame): Feature matrix.
            y (pd.Series): Target vector.

        Returns:
            Tuple[str, Any, float]: The best feature, the best threshold, and the best score.
        """
        best_feature = None
        best_threshold = None
        best_score = -float('inf') if self.criterion in ["information_gain", "gini_index"] else float('inf')

        for feature in X.columns:
            if pd.api.types.is_numeric_dtype(X[feature]):
                thresholds = X[feature].unique()
                for threshold in thresholds:
                    left_y, right_y = self.split_data(X, y, feature, threshold)
                    score = self._calculate_score(left_y, right_y)
                    if (self.criterion in ["information_gain", "gini_index"] and score > best_score) or \
                       (self.criterion == "mse" and score < best_score):
                        best_feature = feature
                        best_threshold = threshold
                        best_score = score
            else:
                categories = X[feature].unique()
                for category in categories:
                    left_y, right_y = self.split_data(X, y, feature, category)
                    score = self._calculate_score(left_y, right_y)
                    if (self.criterion in ["information_gain", "gini_index"] and score > best_score) or \
                       (self.criterion == "mse" and score < best_score):
                        best_feature = feature
                        best_threshold = category
                        best_score = score

        return best_feature, best_threshold, best_score

    def split_data(self, X: pd.DataFrame, y: pd.Series, attribute: str, value: Any) -> Tuple[pd.Series, pd.Series]:
        """
        Splits the data according to an attribute and a given value.

        Args:
            X (pd.DataFrame): Feature matrix.
            y (pd.Series): Target vector.
            attribute (str): The attribute/feature to split on.
            value (Any): The value to split on.

        Returns:
            Tuple[pd.Series, pd.Series]: The split target vectors for the left and right branches.
        """
        if pd.api.types.is_numeric_dtype(X[attribute]):
            left_mask = X[attribute] <= value
            right_mask = X[attribute] > value
        else:
            left_mask = X[attribute] == value
            right_mask = X[attribute] != value
        
        return y[left_mask], y[right_mask]

    def _calculate_score(self, left_y: pd.Series, right_y: pd.Series) -> float:
        """
        Calculates the score of a split based on the criterion.

        Args:
            left_y (pd.Series): Target vector for the left branch.
            right_y (pd.Series): Target vector for the right branch.

        Returns:
            float: The score for the split.
        """
        if self.criterion == "information_gain":
            return self._information_gain(left_y, right_y)
        elif self.criterion == "gini_index":
            return self._gini_index(left_y, right_y)
        elif self.criterion == "mse":
            return self._mean_squared_error(left_y, right_y)

    def _information_gain(self, left_y: pd.Series, right_y: pd.Series) -> float:
        """
        Computes the information gain of a split.

        Args:
            left_y (pd.Series): Target vector for the left branch.
            right_y (pd.Series): Target vector for the right branch.

        Returns:
            float: The information gain of the split.
        """
        total_length = len(left_y) + len(right_y)
        total_entropy = entropy(pd.concat([left_y, right_y]))
    
        left_weight = len(left_y) / total_length
        right_weight = len(right_y) / total_length
        weighted_entropy = (left_weight * entropy(left_y)) + (right_weight * entropy(right_y))
        
        return total_entropy - weighted_entropy

    def _gini_index(self, left_y: pd.Series, right_y: pd.Series) -> float:
        """
        Computes the Gini index of a split.

        Args:
            left_y (pd.Series): Target vector for the left branch.
            right_y (pd.Series): Target vector for the right branch.

        Returns:
            float: The Gini index of the split.
        """
        total_length = len(left_y) + len(right_y)
        left_weight = len(left_y) / total_length
        right_weight = len(right_y) / total_length
        return (left_weight * gini_index(left_y)) + (right_weight * gini_index(right_y))

    def _mean_squared_error(self, left_y: pd.Series, right_y: pd.Series) -> float:
        """
        Computes the mean squared error (MSE) of a split.

        Args:
            left_y (pd.Series): Target vector for the left branch.
            right_y (pd.Series): Target vector for the right branch.

        Returns:
            float: The mean squared error of the split.
        """
        return (mean_squared_error(left_y) + mean_squared_error(right_y)) / 2

    def _most_common_label(self, y: pd.Series) -> Any:
        """
        Returns the most common label in the series.

        Args:
            y (pd.Series): Target vector.

        Returns:
            Any: The most common label.
        """
        return y.mode()[0]

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predicts the class labels or values for the provided data.

        Args:
            X (Union[np.ndarray, pd.DataFrame]): Feature matrix.

        Returns:
            np.ndarray: The predicted labels or values.
        """
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        
        return np.array([self._predict(sample, self.tree_) for _, sample in X.iterrows()])

    def _predict(self, sample: pd.Series, tree: Any) -> Any:
        """
        Traverses the tree to get the prediction for a single sample.

        Args:
            sample (pd.Series): A single data point.
            tree (Any): The structure of the tree node or the label/value if a leaf node is reached.

        Returns:
            Any: The predicted label or value.
        """
        if isinstance(tree, tuple):
            feature, threshold, left_tree, right_tree = tree
            if pd.api.types.is_numeric_dtype(sample[feature]):
                if sample[feature] <= threshold:
                    return self._predict(sample, left_tree)
                else:
                    return self._predict(sample, right_tree)
            else:
                if sample[feature] == threshold:
                    return self._predict(sample, left_tree)
                else:
                    return self._predict(sample, right_tree)
        else:
            return tree

    def plot(self) -> None:
        """
        Plots the structure of the decision tree.

        Returns:
            None
        """
        # Implement a tree visualization method here
        pass

    def get_params(self, deep: bool = True) -> dict:
        """
        Returns the parameters of the DecisionTree object.

        Args:
            deep (bool): Whether to return the deep copy of parameters.

        Returns:
            dict: The parameters of the DecisionTree object.
        """
        return {"criterion": self.criterion, "max_depth": self.max_depth}

    def set_params(self, **params) -> None:
        """
        Sets the parameters of the DecisionTree object.

        Args:
            params (dict): Parameters to set.

        Returns:
            None
        """
        for key, value in params.items():
            setattr(self, key, value)
