from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Initialize Decision Tree classifier
clf = DecisionTreeClassifier(random_state=42)

# Perform 5-fold cross-validation
scores = cross_val_score(clf, X, y, cv=5)

# Print results
print("Cross-validation scores:", scores)
print("Mean accuracy:", scores.mean())