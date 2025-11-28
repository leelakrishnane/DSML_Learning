from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# load the iris dataset as an example
krishna = load_iris()
# print(krishna)
X = krishna.data
y = krishna.target

# split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# fit a decision tree model to the training data
tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)

plt.figure(figsize=(30, 15))
plot_tree(tree, filled=True, feature_names=krishna.feature_names, class_names=krishna.target_names)
plt.show()

"""

### Evaluating the Performance of a Decision Tree Model
Once a decision tree model has been fit to data, we need to evaluate its performance to determine how well it is able to make predictions. There are several metrics that can be used to evaluate the performance of a decision tree model, including accuracy, precision, recall, and the F1-score.

- **Accuracy:** The fraction of correct predictions made by the model.

- **Confusion matrix:** A table that summarizes the number of true positive, true negative, false positive, and false negative predictions made by the model.

- **Precision:** The fraction of positive predictions that are actually positive.

- **Recall:** The fraction of actual positive cases that were correctly predicted as positive by the model.

- **F1-score:** The harmonic mean of precision and recall.

** classification_report **

"""

# make predictions on the test set
y_pred = tree.predict(X_test)

# calculate the accuracy, precision, recall, and F1-score
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
confusion_matrix = confusion_matrix(y_test, y_pred)

# print the results
print(f"Accuracy: {accuracy}")
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print(confusion_matrix)


