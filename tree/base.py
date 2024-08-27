import numpy as np
import pandas as pd
from typing import Literal, Union, Tuple, Any
from dataclasses import dataclass
from sklearn.preprocessing import OneHotEncoder
from .utils import entropy, gini_index, mean_squared_error

@dataclass
class DecisionTree:
    """
    A Decision Tree classifier/regressor that supports splitting based on information gain,
    Gini index, or mean squared error (MSE) for both classification and regression tasks.

    Attributes:
        criterion (Literal["information_gain", "gini_index", "mse"]): The function to measure the quality of a split.
        max_depth (int): The maximum depth of the tree.
        tree_ (Any): The learned decision tree.
        encoder (OneHotEncoder or None): One-hot encoder for categorical feature encoding.
        feature_names (list or None): List of feature names.
    """
    criterion: Literal["information_gain", "gini_index", "mse"]
    max_depth: int

    def __init__(self, criterion: Literal["information_gain", "gini_index", "mse"], max_depth: int = 10):
        """
        Initializes the DecisionTree with the specified criterion and maximum depth.

        Args:
            criterion (Literal["information_gain", "gini_index", "mse"]): The criterion used for splitting.
            max_depth (int, optional): The maximum depth of the tree. Defaults to 10.
        """
        self.criterion = criterion
        self.max_depth = max_depth
        self.tree_ = None
        self.encoder = None
        self.feature_names = None

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> None:
        """
        Fits the decision tree model to the provided data.

        Args:
            X (Union[np.ndarray, pd.DataFrame]): The input features, which can be a NumPy array or pandas DataFrame.
            y (Union[np.ndarray, pd.Series]): The target labels, which can be a NumPy array or pandas Series.
        
        Algorithm:
            1. Convert input data X and y to pandas DataFrame and Series, respectively, if they are NumPy arrays.
            2. One-hot encode categorical features, if any, and concatenate with the rest of the data.
            3. Initialize tree building with the recursive helper function `_fit`.
        """
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        if isinstance(y, np.ndarray):
            y = pd.Series(y)

        X.columns = X.columns.astype(str)
        
        # Identify categorical features and apply one-hot encoding
        categorical_cols = X.select_dtypes(include=['category']).columns
        if len(categorical_cols) > 0:
            self.encoder = OneHotEncoder(drop='first', sparse_output=False)
            X_encoded = pd.DataFrame(self.encoder.fit_transform(X[categorical_cols]), 
                                     columns=self.encoder.get_feature_names_out(categorical_cols))
            X = pd.concat([X.drop(columns=categorical_cols), X_encoded], axis=1)

        self.feature_names = X.columns.tolist()
        self.X_ = X
        self.y_ = y
        self.n_features_ = X.shape[1]
        self.tree_ = self._fit(X, y, depth=0)

    def _fit(self, X: pd.DataFrame, y: pd.Series, depth: int) -> Any:
        """
        Recursively builds the decision tree using the training data.

        Args:
            X (pd.DataFrame): The input features as a DataFrame.
            y (pd.Series): The target labels as a Series.
            depth (int): The current depth of the tree.

        Returns:
            Any: The subtree or leaf node value.

        Algorithm:
            1. If all target labels are the same, return the label (pure node).
            2. If the maximum depth is reached or there are no features left, return the most common label.
            3. Find the best feature and threshold to split the data using `_best_split`.
            4. Split the data into left and right subsets and recursively call `_fit` on each.
            5. Return a tuple representing the decision node.
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
        
        left_indices = X[best_feature] <= best_threshold
        right_indices = X[best_feature] > best_threshold

        if left_indices.sum() == 0 or right_indices.sum() == 0:
            return self._most_common_label(y)

        left_tree = self._fit(X[left_indices], y[left_indices], depth + 1)
        right_tree = self._fit(X[right_indices], y[right_indices], depth + 1)
        
        return (best_feature, best_threshold, left_tree, right_tree)

    def _predict(self, sample: pd.Series, tree: Any) -> Any:
        """
        Predicts the label for a single sample using the trained decision tree.

        Args:
            sample (pd.Series): A single sample as a Series.
            tree (Any): The trained decision tree.

        Returns:
            Any: The predicted label.

        Algorithm:
            1. If the current tree node is a leaf, return the label.
            2. Compare the sample's feature value with the threshold and recursively predict using the left or right subtree.
        """
        if not isinstance(tree, tuple):
            return tree
        
        feature, threshold, left_tree, right_tree = tree
        if sample[feature] <= threshold:
            return self._predict(sample, left_tree)
        else:
            return self._predict(sample, right_tree)

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predicts labels for the given input data using the trained decision tree.

        Args:
            X (Union[np.ndarray, pd.DataFrame]): The input features to predict labels for.

        Returns:
            np.ndarray: Predicted labels for each sample in the input data.

        Algorithm:
            1. Convert input data X to a DataFrame if it is a NumPy array.
            2. Apply one-hot encoding to categorical features if the model was trained with encoded features.
            3. For each sample, call `_predict` to get the predicted label.
        """
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        
        # If the model was trained with one-hot encoded features, apply the same transformation to the input data
        if self.encoder:
            categorical_cols = X.select_dtypes(include=['category']).columns
            if len(categorical_cols) > 0:
                X_encoded = pd.DataFrame(self.encoder.transform(X[categorical_cols]), 
                                         columns=self.encoder.get_feature_names_out(categorical_cols))
                X = pd.concat([X.drop(columns=categorical_cols), X_encoded], axis=1)

        return np.array([self._predict(sample, self.tree_) for _, sample in X.iterrows()])

    def _best_split(self, X: pd.DataFrame, y: pd.Series) -> Tuple[str, Any, float]:
        """
        Finds the best feature and threshold to split the data on, based on the chosen criterion.

        Args:
            X (pd.DataFrame): The input features as a DataFrame.
            y (pd.Series): The target labels as a Series.

        Returns:
            Tuple[str, Any, float]: The best feature, threshold, and the corresponding score.

        Algorithm:
            1. Initialize variables to store the best feature, threshold, and score.
            2. For each feature, compute possible thresholds (for numerical features) or categories (for categorical features).
            3. For each threshold/category, split the data and compute the score using `_calculate_score`.
            4. Update the best feature, threshold, and score if the current split is better.
            5. Return the best feature, threshold, and score.
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
        Splits the target labels into two groups based on the specified feature and its value.

        Args:
            X (pd.DataFrame): The input features as a DataFrame.
            y (pd.Series): The target labels as a Series.
            attribute (str): The feature used for splitting.
            value (Any): The value or threshold for splitting.

        Returns:
            Tuple[pd.Series, pd.Series]: The target labels for the left and right splits.

        Algorithm:
            1. If the attribute is numeric, split the data based on whether it is less than or equal to the threshold.
            2. If the attribute is categorical, split the data based on whether it matches the category.
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
        Calculates the score (information gain, Gini index, or MSE) for a split.

        Args:
            left_y (pd.Series): The target labels for the left split.
            right_y (pd.Series): The target labels for the right split.

        Returns:
            float: The calculated score for the split.

        Algorithm:
            1. Depending on the chosen criterion, calculate the score using the corresponding method.
        """
        if self.criterion == "information_gain":
            return self._information_gain(left_y, right_y)
        elif self.criterion == "gini_index":
            return self._gini_index(left_y, right_y)
        elif self.criterion == "mse":
            return self._mean_squared_error(left_y, right_y)

    def _information_gain(self, left_y: pd.Series, right_y: pd.Series) -> float:
        """
        Computes the information gain for a given split.

        Args:
            left_y (pd.Series): The target labels for the left split.
            right_y (pd.Series): The target labels for the right split.

        Returns:
            float: The information gain value.

        Algorithm:
            1. Calculate the total entropy of the parent node.
            2. Compute the weighted entropy of the left and right children.
            3. Subtract the weighted entropy from the total entropy to get the information gain.
        """
        total_length = len(left_y) + len(right_y)
        total_entropy = entropy(pd.concat([left_y, right_y]))
    
        left_weight = len(left_y) / total_length
        right_weight = len(right_y) / total_length
        weighted_entropy = (left_weight * entropy(left_y)) + (right_weight * entropy(right_y))
        
        return total_entropy - weighted_entropy

    def _gini_index(self, left_y: pd.Series, right_y: pd.Series) -> float:
        """
        Computes the Gini index for a given split.

        Args:
            left_y (pd.Series): The target labels for the left split.
            right_y (pd.Series): The target labels for the right split.

        Returns:
            float: The Gini index value.

        Algorithm:
            1. Calculate the proportion of samples in the left and right splits.
            2. Compute the Gini index for each split and combine them using their proportions.
        """
        p_left = len(left_y) / (len(left_y) + len(right_y))
        p_right = 1 - p_left
        return gini_index(left_y) * p_left + gini_index(right_y) * p_right

    def _mean_squared_error(self, left_y: pd.Series, right_y: pd.Series) -> float:
        """
        Computes the mean squared error (MSE) for a given split.

        Args:
            left_y (pd.Series): The target labels for the left split.
            right_y (pd.Series): The target labels for the right split.

        Returns:
            float: The MSE value.

        Algorithm:
            1. Calculate the MSE for both left and right splits.
            2. Compute the weighted average of the MSE values based on the number of samples in each split.
        """
        return (len(left_y) * mean_squared_error(left_y) + len(right_y) * mean_squared_error(right_y)) / (len(left_y) + len(right_y))

    def _most_common_label(self, y: pd.Series) -> Any:
        """
        Returns the most common label in the provided target labels.

        Args:
            y (pd.Series): The target labels.

        Returns:
            Any: The most frequent label.

        Algorithm:
            1. Use the pandas `value_counts` method to determine the most common label.
        """
        return y.value_counts().idxmax()

    def plot(self) -> None:
        """
        Prints a visual representation of the decision tree.

        Algorithm:
            1. Recursively traverse the tree and print each node.
            2. For each node, print the feature and threshold used for splitting.
            3. For leaf nodes, print the class label.
        """
        def plot_tree(tree, feature_names, depth=0):
            if not isinstance(tree, tuple):
                print(f"{'  ' * depth}Class: {tree}")
                return

            feature, threshold, left_tree, right_tree = tree
            feature_name = feature_names[feature] if feature in feature_names else str(feature)
            print(f"{'  ' * depth}?({feature_name} <= {threshold})")
            print(f"{'  ' * (depth + 1)}Y: ", end="")
            plot_tree(left_tree, feature_names, depth + 1)
            print(f"{'  ' * (depth + 1)}N: ", end="")
            plot_tree(right_tree, feature_names, depth + 1)

        if self.tree_ is None:
            print("The tree is not yet trained!")
            return

        feature_names = {col: name for col, name in enumerate(self.X_.columns.tolist())}
        plot_tree(self.tree_, feature_names)

    def get_params(self, deep=True) -> dict:
        """
        Gets the parameters of the decision tree model.

        Args:
            deep (bool): If True, returns deep copy of parameters.

        Returns:
            dict: The model parameters.
        """
        return {"criterion": self.criterion, "max_depth": self.max_depth}

    def set_params(self, **params) -> 'DecisionTree':
        """
        Sets the parameters of the decision tree model.

        Args:
            **params: Arbitrary keyword arguments representing model parameters.

        Returns:
            DecisionTree: The updated model instance.

        Algorithm:
            1. Loop through the provided parameters and set them using `setattr`.
        """
        for param, value in params.items():
            setattr(self, param, value)
        return self
