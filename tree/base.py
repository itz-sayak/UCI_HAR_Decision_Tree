"""

This module defines the `DecisionTree` class, implementing a decision tree algorithm
for classification tasks. The `DecisionTree` class supports training, predicting, and visualizing
the decision tree. It handles numerical and categorical features and provides various
criteria for splitting nodes.

Imports:
    - numpy (np): Library for numerical operations.
    - pandas (pd): Library for data manipulation and analysis.
    - typing: Provides type hinting for variables and function signatures.
    - dataclasses: Provides a decorator to generate special methods for classes automatically.

Classes:
    DecisionTree:
        Attributes:
            criterion (Literal["information_gain", "gini_index"]): Criterion for splitting nodes.
                Can be "information_gain" or "gini_index".
            max_depth (int): Maximum depth of the tree. Defaults to 5.
            tree_ (Any): The trained decision tree. Initially, it is `None` and set after training.

        Methods:
            __init__(self, criterion: Literal["information_gain", "gini_index"], max_depth: int = 5):
                Initializes the decision tree with the given criterion and maximum depth.

            fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> None:
                Trains the decision tree using the input features `X` and target values `y`.

            _fit(self, X: pd.DataFrame, y: pd.Series, depth: int) -> Any:
                Recursive function to construct the decision tree. Used internally by `fit`.

            _best_split(self, X: pd.DataFrame, y: pd.Series) -> Tuple[str, Any, float]:
                Determines the best feature and threshold for splitting the data based on the chosen criterion.

            split_data(self, X: pd.DataFrame, y: pd.Series, attribute: str, value: Any) -> Tuple[pd.Series, pd.Series]:
                Splits the data into two subsets based on the given attribute and value.

            _calculate_score(self, left_y: pd.Series, right_y: pd.Series) -> float:
                Calculates the score for a split based on the criterion.

            _information_gain(self, left_y: pd.Series, right_y: pd.Series) -> float:
                Computes the information gain from a split.

            _gini_index(self, left_y: pd.Series, right_y: pd.Series) -> float:
                Computes the Gini index from a split.

            _mean_squared_error(self, left_y: pd.Series, right_y: pd.Series) -> float:
                Computes the mean squared error (MSE) from a split.

            _most_common_label(self, y: pd.Series) -> Any:
                Returns the most common label in the series.

            predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
                Predicts class labels for the provided data.

            _predict(self, sample: pd.Series, tree: Any) -> Any:
                Traverses the decision tree to get the prediction for a single sample.

            plot(self) -> None:
                Prints a textual representation of the decision tree.

            get_params(self, deep=True) -> dict:
                Returns the parameters of the decision tree.

            set_params(self, **params) -> 'DecisionTree':
                Sets the parameters of the decision tree.

Usage Example:
    from tree.base import DecisionTree

    # Initialize and train the decision tree
    dt = DecisionTree(criterion="gini_index", max_depth=3)
    dt.fit(X_train, y_train)

    # Predict using the trained tree
    predictions = dt.predict(X_test)

    # Plot the decision tree
    dt.plot()

Notes:
    - The `fit` method preprocesses the input data and constructs the tree using recursive methods.
    - The `predict` method traverses the constructed tree to generate predictions for new data.
    - The `plot` method provides a basic textual representation of the tree structure.
"""


import numpy as np
import pandas as pd
from typing import Literal, Union, Tuple, Any
from dataclasses import dataclass
from .utils import entropy, gini_index, mean_squared_error

@dataclass
class DecisionTree:
    criterion: Literal["information_gain", "gini_index"]
    max_depth: int

    def __init__(self, criterion: Literal["information_gain", "gini_index"], max_depth: int = 5):
        self.criterion = criterion
        self.max_depth = max_depth
        self.tree_ = None

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> None:
        """
        Function to train and construct the decision tree
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
        Recursive function to construct the tree.
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
        Function to find the best split based on the criterion.
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
        Function to split the data according to an attribute.
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
        Function to calculate the score for the split
        """
        if self.criterion == "information_gain":
            return self._information_gain(left_y, right_y)
        elif self.criterion == "gini_index":
            return self._gini_index(left_y, right_y)
        elif self.criterion == "mse":
            return self._mean_squared_error(left_y, right_y)

    def _information_gain(self, left_y: pd.Series, right_y: pd.Series) -> float:
        """
        Function to calculate the information gain
        """
        total_length = len(left_y) + len(right_y)
        total_entropy = entropy(pd.concat([left_y, right_y]))
    
        left_weight = len(left_y) / total_length
        right_weight = len(right_y) / total_length
        weighted_entropy = (left_weight * entropy(left_y)) + (right_weight * entropy(right_y))
        
        return total_entropy - weighted_entropy

    def _gini_index(self, left_y: pd.Series, right_y: pd.Series) -> float:
        """
        Function to calculate the Gini index
        """
        p_left = len(left_y) / (len(left_y) + len(right_y))
        p_right = 1 - p_left
        return gini_index(left_y) * p_left + gini_index(right_y) * p_right

    def _mean_squared_error(self, left_y: pd.Series, right_y: pd.Series) -> float:
        """
        Function to calculate the mean squared error (MSE)
        """
        return (len(left_y) * mean_squared_error(left_y) + len(right_y) * mean_squared_error(right_y)) / (len(left_y) + len(right_y))

    def _most_common_label(self, y: pd.Series) -> Any:
        """
        Function to return the most common label in the series
        """
        return y.value_counts().idxmax()

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict the class labels for the provided data.
        """
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        
        return np.array([self._predict(sample, self.tree_) for _, sample in X.iterrows()])

    def _predict(self, sample: pd.Series, tree: Any) -> Any:
        """
        Traverse the tree to get the prediction for a single sample.
        """
        if not isinstance(tree, tuple):
            return tree
        
        feature, threshold, left_tree, right_tree = tree
        if sample[feature] <= threshold:
            return self._predict(sample, left_tree)
        else:
            return self._predict(sample, right_tree)

    def plot(self) -> None:
        """
        Function to plot the tree
        """
        def plot_tree(tree, feature_names, depth=0):
            if not isinstance(tree, tuple):
                print(f"{'  ' * depth}Class: {tree}")
                return

            feature, threshold, left_tree, right_tree = tree
            feature_name = feature_names[feature]
            print(f"{'  ' * depth}?({feature_name} <= {threshold})")
            print(f"{'  ' * (depth + 1)}Y: ", end="")
            plot_tree(left_tree, feature_names, depth + 1)
            print(f"{'  ' * (depth + 1)}N: ", end="")
            plot_tree(right_tree, feature_names, depth + 1)
        
        if self.tree_ is None:
            print("The tree is not yet trained!")
            return
        
        feature_names = {i: name for i, name in enumerate(self.X_.columns.tolist())}
        plot_tree(self.tree_, feature_names)

    def get_params(self, deep=True) -> dict:
        """
        Get parameters for this estimator.
        """
        return {"criterion": self.criterion, "max_depth": self.max_depth}

    def set_params(self, **params) -> 'DecisionTree':
        """
        Set the parameters of this estimator.
        """
        for param, value in params.items():
            setattr(self, param, value)
        return self
