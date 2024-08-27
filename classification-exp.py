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

#(a)

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

#(b)

k = 5

# Initialize lists to store outer fold results
outer_fold_accuracies = []
outer_fold_precisions = []
outer_fold_recalls = []

# Calculate the size of each outer fold
fold_size = len(X) // k

# Perform k-fold cross-validation (outer loop)
for i in range(k):
    test_start = i * fold_size
    test_end = (i + 1) * fold_size
    test_set = X[test_start:test_end]
    test_labels = y[test_start:test_end]
    
    training_set = np.concatenate((X[:test_start], X[test_end:]), axis=0)
    training_labels = np.concatenate((y[:test_start], y[test_end:]), axis=0)
    
    print(f"Outer Fold {i + 1} - Training Shape: {training_set.shape}")

    # Nested Cross-Validation to find the best max_depth (inner loop)
    best_depth = None
    best_accuracy = -np.inf
    
    for depth in range(1, 11):  # Checking depths from 1 to 10
        inner_fold_accuracies = []
        
        # Split training data into 5 inner folds
        inner_fold_size = len(training_set) // k
        for j in range(k):
            val_start = j * inner_fold_size
            val_end = (j + 1) * inner_fold_size
            
            inner_val_set = training_set[val_start:val_end]
            inner_val_labels = training_labels[val_start:val_end]
            
            inner_train_set = np.concatenate((training_set[:val_start], training_set[val_end:]), axis=0)
            inner_train_labels = np.concatenate((training_labels[:val_start], training_labels[val_end:]), axis=0)
            
            # Train model for inner fold
            dt_classifier = DecisionTree(criterion='gini_index', max_depth=depth)
            dt_classifier.fit(pd.DataFrame(inner_train_set, columns=['Feature1', 'Feature2']), pd.Series(inner_train_labels))
            
            # Evaluate on inner validation set
            inner_val_df = pd.DataFrame(inner_val_set, columns=['Feature1', 'Feature2'])
            inner_predictions = dt_classifier.predict(inner_val_df)
            
            inner_accuracy = accuracy(pd.Series(inner_predictions), pd.Series(inner_val_labels))
            inner_fold_accuracies.append(inner_accuracy)
        
        # Calculate the average accuracy for this depth
        avg_inner_accuracy = np.mean(inner_fold_accuracies)
        if avg_inner_accuracy > best_accuracy:
            best_accuracy = avg_inner_accuracy
            best_depth = depth
    
    print(f"Best depth for Outer Fold {i + 1}: {best_depth} with accuracy: {best_accuracy:.4f}")

    # Train on the full outer training set with the best depth
    final_dt_classifier = DecisionTree(criterion='gini_index', max_depth=best_depth)
    final_dt_classifier.fit(pd.DataFrame(training_set, columns=['Feature1', 'Feature2']), pd.Series(training_labels))
    
    # Evaluate on the outer test set
    test_df = pd.DataFrame(test_set, columns=['Feature1', 'Feature2'])
    test_predictions = final_dt_classifier.predict(test_df)
    
    outer_accuracy = accuracy(pd.Series(test_predictions), pd.Series(test_labels))
    outer_precisions = {cls: precision(pd.Series(test_predictions), pd.Series(test_labels), cls) for cls in np.unique(test_labels)}
    outer_recalls = {cls: recall(pd.Series(test_predictions), pd.Series(test_labels), cls) for cls in np.unique(test_labels)}
    
    outer_fold_accuracies.append(outer_accuracy)
    outer_fold_precisions.append(outer_precisions)
    outer_fold_recalls.append(outer_recalls)

# Print final results
for i in range(k):
    print(f"Outer Fold {i + 1} - Accuracy: {outer_fold_accuracies[i]:.4f}")
    for cls, prec in outer_fold_precisions[i].items():
        print(f'  Class {cls} - Precision: {prec:.4f}')
    for cls, rec in outer_fold_recalls[i].items():
        print(f'  Class {cls} - Recall: {rec:.4f}')
