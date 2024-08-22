import pandas as pd
import numpy as np
from tree.base import DecisionTree
from metrics import *
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

np.random.seed(42)

# Code given in the question
X, y = make_classification(
    n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define parameter grid for depth
param_grid = {'max_depth': np.arange(1, 11)}

# Initialize the GridSearchCV object
grid_search = GridSearchCV(
    estimator=DecisionTree(criterion='gini_index'), 
    param_grid=param_grid, 
    scoring='accuracy', 
    cv=5
)

# Perform the search
grid_search.fit(X_train, y_train)

# Get the best parameters and results
best_depth = grid_search.best_params_['max_depth']
print(f'Optimal tree depth: {best_depth}')

# Train the best model and evaluate
best_tree = DecisionTree(criterion='gini_index', max_depth=best_depth)
best_tree.fit(pd.DataFrame(X_train, columns=['Feature1', 'Feature2']), pd.Series(y_train))
y_pred = best_tree.predict(pd.DataFrame(X_test, columns=['Feature1', 'Feature2']))

# Evaluate the best model
acc = accuracy(pd.Series(y_pred), pd.Series(y_test))
precisions = {cls: precision(pd.Series(y_pred), pd.Series(y_test), cls) for cls in np.unique(y_test)}
recalls = {cls: recall(pd.Series(y_pred), pd.Series(y_test), cls) for cls in np.unique(y_test)}

print(f'Accuracy with optimal depth: {acc:.2f}')
print('Precision:')
for cls, prec in precisions.items():
    print(f'  Class {cls}: {prec:.2f}')
print('Recall:')
for cls, rec in recalls.items():
    print(f'  Class {cls}: {rec:.2f}')
