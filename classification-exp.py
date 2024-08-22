import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import accuracy, precision, recall
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

np.random.seed(42)

# Code given in the question
X, y = make_classification(
    n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5)

# Plot the dataset
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Generated Classification Dataset')
plt.show()

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the Decision Tree
tree = DecisionTree(criterion='gini_index', max_depth=5)
tree.fit(pd.DataFrame(X_train, columns=['Feature1', 'Feature2']), pd.Series(y_train))

# Predict on the test set
y_pred = tree.predict(pd.DataFrame(X_test, columns=['Feature1', 'Feature2']))

# Evaluate the model
acc = accuracy(pd.Series(y_pred), pd.Series(y_test))
precisions = {cls: precision(pd.Series(y_pred), pd.Series(y_test), cls) for cls in np.unique(y_test)}
recalls = {cls: recall(pd.Series(y_pred), pd.Series(y_test), cls) for cls in np.unique(y_test)}

print(f'Accuracy: {acc:.2f}')
print('Precision:')
for cls, prec in precisions.items():
    print(f'  Class {cls}: {prec:.2f}')
print('Recall:')
for cls, rec in recalls.items():
    print(f'  Class {cls}: {rec:.2f}')
