import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *

np.random.seed(42)

# Test Case 1: Real Input and Real Output
print("\nTest case 1--> Real Input and Real Output")
N = 30
P = 5
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randn(N))

print("\n" + "-"*50)  # Separator line
print(f"Criteria : mse\n")

# Use MSE for real outputs
tree = DecisionTree(criterion="mse")  # Initialize tree with MSE criterion
tree.fit(X, y)  # Fit the tree with data

y_hat = tree.predict(X)  # Make predictions

tree.plot()  # Plot the decision tree

print(f"\nRMSE: {rmse(y_hat, y)}")
print(f"MAE: {mae(y_hat, y)}\n")


# Test Case 2: Real Input and Discrete Output
print("\nTest case 2--> Real Input and Discrete Output")
N = 30
P = 5
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randint(P, size=N), dtype="category")

for criteria in ["information_gain", "gini_index"]:
    print("\n" + "-"*50)  # Separator line
    print(f"Criteria : {criteria}\n")
    
    tree = DecisionTree(criterion=criteria)  # Initialize tree with criterion
    tree.fit(X, y)  # Fit the tree with data
    
    y_hat = tree.predict(X)  # Make predictions
    
    tree.plot()  # Plot the decision tree
    
    print(f"\nAccuracy: {accuracy(y_hat, y)}")
    for cls in y.unique():
        print(f"\nCLASS: {cls}")
        print(f"Precision: {precision(y_hat, y, cls)}")
        print(f"Recall: {recall(y_hat, y, cls)}\n")


# Test Case 3: Discrete Input and Discrete Output
print("\nTest case 3--> Discrete Input and Discrete Output")
N = 30
P = 5
X = pd.DataFrame({i: pd.Series(np.random.randint(P, size=N), dtype="category") for i in range(5)})
y = pd.Series(np.random.randint(P, size=N), dtype="category")

for criteria in ["information_gain", "gini_index"]:
    print("\n" + "-"*50)  # Separator line
    print(f"Criteria : {criteria}\n")
    
    tree = DecisionTree(criterion=criteria)  # Initialize tree with criterion
    tree.fit(X, y)  # Fit the tree with data
    
    y_hat = tree.predict(X)  # Make predictions
    
    tree.plot()  # Plot the decision tree
    
    print(f"\nAccuracy: {accuracy(y_hat, y)}")
    for cls in y.unique():
        print(f"\nCLASS: {cls}")
        print(f"Precision: {precision(y_hat, y, cls)}")
        print(f"Recall: {recall(y_hat, y, cls)}\n")


# Test Case 4: Discrete Input and Real Output
print("\nTest case 4--> Discrete Input and Real Output")
N = 30
P = 5
X = pd.DataFrame({i: pd.Series(np.random.randint(P, size=N), dtype="category") for i in range(5)})
y = pd.Series(np.random.randn(N))

print("\n" + "-"*50)  # Separator line
print(f"Criteria : mse\n")

# Use MSE for real outputs
tree = DecisionTree(criterion="mse")  # Initialize tree with MSE criterion
tree.fit(X, y)  # Fit the tree with data

y_hat = tree.predict(X)  # Make predictions

tree.plot()  # Plot the decision tree

print(f"\nRMSE: {rmse(y_hat, y)}")
print(f"MAE: {mae(y_hat, y)}\n")
