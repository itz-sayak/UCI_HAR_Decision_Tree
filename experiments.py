import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

np.random.seed(42)
num_average_time = 100  # Number of times to run each experiment to calculate the average values

def measure_time(model, X_train, X_test, y_train, y_test):
    fit_times = []
    predict_times = []
    
    for _ in range(num_average_time):
        start_time = time.time()
        model.fit(X_train, y_train)
        fit_times.append(time.time() - start_time)
        
        start_time = time.time()
        y_pred = model.predict(X_test)
        predict_times.append(time.time() - start_time)
    
    avg_fit_time = np.mean(fit_times)
    avg_predict_time = np.mean(predict_times)
    
    return avg_fit_time, avg_predict_time

# Load and preprocess data similarly to auto-efficiency.py
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
data = pd.read_csv(url, delim_whitespace=True, header=None,
                 names=["mpg", "cylinders", "displacement", "horsepower", "weight",
                        "acceleration", "model year", "origin", "car name"])

data = data.replace('?', np.nan).dropna()
data["horsepower"] = data["horsepower"].astype(float)
data["origin"] = data["origin"].astype('category').cat.codes
data = data.drop(["car name"], axis=1)

X = data.drop("mpg", axis=1)
y = (data["mpg"] > data["mpg"].median()).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Measure performance of custom DecisionTree
custom_dt = DecisionTree(criterion="gini_index", max_depth=5)
custom_fit_time, custom_predict_time = measure_time(custom_dt, X_train, X_test, y_train, y_test)

# Measure performance of scikit-learn DecisionTreeClassifier
sklearn_dt = DecisionTreeClassifier(criterion="gini", max_depth=5, random_state=42)
sklearn_fit_time, sklearn_predict_time = measure_time(sklearn_dt, X_train, X_test, y_train, y_test)

# Plotting the results
labels = ['Fit Time', 'Predict Time']
custom_times = [custom_fit_time, custom_predict_time]
sklearn_times = [sklearn_fit_time, sklearn_predict_time]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, custom_times, width, label='Custom DT')
rects2 = ax.bar(x + width/2, sklearn_times, width, label='Scikit-learn DT')

ax.set_xlabel('Metrics')
ax.set_ylabel('Time (seconds)')
ax.set_title('Comparison of Fit and Predict Times')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

fig.tight_layout()

# Save the plot as an image file
plt.savefig('/content/comparison_plot.png', format='png')

# Optionally, display the plot
plt.show()
