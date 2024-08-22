import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

np.random.seed(42)

# Reading the data
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
data = pd.read_csv(url, delim_whitespace=True, header=None,
                 names=["mpg", "cylinders", "displacement", "horsepower", "weight",
                        "acceleration", "model year", "origin", "car name"])

# Data Preprocessing
data = data.replace('?', np.nan).dropna()
data["horsepower"] = data["horsepower"].astype(float)
data["origin"] = data["origin"].astype('category').cat.codes  # Convert categorical data to numeric
data = data.drop(["car name"], axis=1)

# Define features and target
X = data.drop("mpg", axis=1)
y = (data["mpg"] > data["mpg"].median()).astype(int)  # Convert mpg to binary classification

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train and evaluate custom DecisionTree
custom_dt = DecisionTree(criterion="gini_index", max_depth=5)
custom_dt.fit(X_train, y_train)
y_pred_custom = custom_dt.predict(X_test)

print("Custom Decision Tree Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_custom)}")
print(f"Precision: {precision_score(y_test, y_pred_custom)}")
print(f"Recall: {recall_score(y_test, y_pred_custom)}")
print(f"F1 Score: {f1_score(y_test, y_pred_custom)}")

# Train and evaluate scikit-learn DecisionTreeClassifier
sklearn_dt = DecisionTreeClassifier(criterion="gini", max_depth=5, random_state=42)
sklearn_dt.fit(X_train, y_train)
y_pred_sklearn = sklearn_dt.predict(X_test)

print("\nScikit-learn Decision Tree Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_sklearn)}")
print(f"Precision: {precision_score(y_test, y_pred_sklearn)}")
print(f"Recall: {recall_score(y_test, y_pred_sklearn)}")
print(f"F1 Score: {f1_score(y_test, y_pred_sklearn)}")

# Plotting comparison
labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
custom_scores = [
    accuracy_score(y_test, y_pred_custom),
    precision_score(y_test, y_pred_custom),
    recall_score(y_test, y_pred_custom),
    f1_score(y_test, y_pred_custom)
]
sklearn_scores = [
    accuracy_score(y_test, y_pred_sklearn),
    precision_score(y_test, y_pred_sklearn),
    recall_score(y_test, y_pred_sklearn),
    f1_score(y_test, y_pred_sklearn)
]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, custom_scores, width, label='Custom DT')
rects2 = ax.bar(x + width/2, sklearn_scores, width, label='Scikit-learn DT')

ax.set_xlabel('Metrics')
ax.set_ylabel('Scores')
ax.set_title('Comparison of Custom and Scikit-learn Decision Trees')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

fig.tight_layout()

# Save the plot as an image file
plt.savefig('/content/Comparison of Decision Trees.png', format='png')

plt.show()
