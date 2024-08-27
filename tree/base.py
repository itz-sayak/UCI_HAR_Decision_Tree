import numpy as np
import pandas as pd
from typing import Literal, Union, Tuple, Any
from dataclasses import dataclass
from sklearn.preprocessing import OneHotEncoder
from .utils import entropy, gini_index, mean_squared_error

@dataclass
class DecisionTree:
    criterion: Literal["information_gain", "gini_index", "mse"]
    max_depth: int

    def __init__(self, criterion: Literal["information_gain", "gini_index", "mse"], max_depth: int = 10):
        self.criterion = criterion
        self.max_depth = max_depth
        self.tree_ = None
        self.encoder = None
        self.feature_names = None

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> None:
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
        if not isinstance(tree, tuple):
            return tree
        
        feature, threshold, left_tree, right_tree = tree
        if sample[feature] <= threshold:
            return self._predict(sample, left_tree)
        else:
            return self._predict(sample, right_tree)

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
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
        if pd.api.types.is_numeric_dtype(X[attribute]):
            left_mask = X[attribute] <= value
            right_mask = X[attribute] > value
        else:
            left_mask = X[attribute] == value
            right_mask = X[attribute] != value
        
        return y[left_mask], y[right_mask]

    def _calculate_score(self, left_y: pd.Series, right_y: pd.Series) -> float:
        if self.criterion == "information_gain":
            return self._information_gain(left_y, right_y)
        elif self.criterion == "gini_index":
            return self._gini_index(left_y, right_y)
        elif self.criterion == "mse":
            return self._mean_squared_error(left_y, right_y)

    def _information_gain(self, left_y: pd.Series, right_y: pd.Series) -> float:
        total_length = len(left_y) + len(right_y)
        total_entropy = entropy(pd.concat([left_y, right_y]))
    
        left_weight = len(left_y) / total_length
        right_weight = len(right_y) / total_length
        weighted_entropy = (left_weight * entropy(left_y)) + (right_weight * entropy(right_y))
        
        return total_entropy - weighted_entropy

    def _gini_index(self, left_y: pd.Series, right_y: pd.Series) -> float:
        p_left = len(left_y) / (len(left_y) + len(right_y))
        p_right = 1 - p_left
        return gini_index(left_y) * p_left + gini_index(right_y) * p_right

    def _mean_squared_error(self, left_y: pd.Series, right_y: pd.Series) -> float:
        return (len(left_y) * mean_squared_error(left_y) + len(right_y) * mean_squared_error(right_y)) / (len(left_y) + len(right_y))

    def _most_common_label(self, y: pd.Series) -> Any:
        return y.value_counts().idxmax()

    def plot(self) -> None:
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
        return {"criterion": self.criterion, "max_depth": self.max_depth}

    def set_params(self, **params) -> 'DecisionTree':
        for param, value in params.items():
            setattr(self, param, value)
        return self
