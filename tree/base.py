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
    Recursively constructs the decision tree.

    This method builds the decision tree by recursively splitting the data based on the best feature and threshold. It operates as follows:

    1. Check if All Labels are the Same:
       - If all the labels in the current subset `y` are the same, the method returns this label as the result for this node. This indicates that the node is a leaf node with a single class.

    2. Check if Maximum Depth is Reached:
       - If the current depth of the tree has reached the specified `max_depth`, the method returns the most common label in `y`. This prevents the tree from growing beyond the specified depth.

    3. Check if Dataset is Empty:
       - If the dataset `X` is empty, the method returns the most common label in `y`. This handles cases where a split results in an empty dataset.

    4. Find the Best Feature and Threshold:
       - The method calls `_best_split` to determine the best feature and threshold (or value) for splitting the data based on the chosen criterion (`criterion`). It evaluates all possible splits and selects the one that provides the best score.

    5. Handle Cases with No Valid Split:
       - If no valid split is found (i.e., no feature provides an improvement), the method returns the most common label in `y`.

    6. Determine How to Split the Data:
       - Depending on whether the best feature is numeric or categorical, the dataset `X` is split into left and right subsets based on the best threshold or category. Numeric features are split using a threshold, while categorical features are split using equality.

    7. Check for Empty Splits:
       - If the left or right subset resulting from the split is empty, the method returns the most common label in `y` to avoid further processing with an empty dataset.

    8. Recursively Build Subtrees:
       - The method recursively calls itself to build the left and right subtrees using the left and right subsets of the data. The `depth` parameter is incremented by 1 for each recursive call to track the depth of the tree.

    9. Return the Current Node:
       - The method returns a tuple representing the current node of the tree. This tuple includes the best feature, threshold for the split, and left and right subtrees.

    Parameters:
    - X (pd.DataFrame): The features of the dataset.
    - y (pd.Series): The labels of the dataset.
    - depth (int): The current depth of the tree.

    Returns:
    - Any: A tuple representing the current node of the tree if further splitting is possible; otherwise, a leaf node with a class label.
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
        Finds the best feature and threshold for splitting the data based on the criterion.

    This method evaluates all possible splits in the dataset to determine the optimal feature and threshold (or value) for partitioning the data. The chosen split is the one that maximizes or minimizes the score, depending on the criterion. The process is as follows:

    1. **Initialize Best Values**:
       - The method initializes variables to keep track of the best feature, best threshold, and best score. The initial best score is set to negative infinity for criteria where higher scores are better (e.g., information gain, Gini index) and positive infinity for criteria where lower scores are better (e.g., mean squared error).

    2. **Iterate Over Features**:
       - The method loops through each feature in the dataset to evaluate potential splits.

    3. **Evaluate Numeric Features**:
       - The method evaluates all unique values in the feature as potential thresholds for numeric features. For each threshold:
         - The data is split into left and right subsets based on the threshold.
         - The score for this split is calculated using the `_calculate_score` method.
         - If the score for the split is better (higher or lower, depending on the criterion) than the current best score, the feature, threshold, and score are updated.

    4. **Evaluate Categorical Features**:
       - The method evaluates each unique category as a potential split point for categorical features. For each category:
         - The data is split into left and right subsets based on the category.
         - The score for this split is calculated using the `_calculate_score` method.
         - If the score for the split is better (higher or lower, depending on the criterion) than the current best score, the feature, category, and score are updated.

    5. **Return Best Split**:
       - The method returns the feature and threshold (or category) that resulted in the best score, along with the best score itself.

    Parameters:
    - X (pd.DataFrame): The features of the dataset.
    - y (pd.Series): The labels of the dataset.

    Returns:
    - Tuple[str, Any, float]: A tuple containing the best feature, the best threshold or category for splitting, and the best score achieved by this split.
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
