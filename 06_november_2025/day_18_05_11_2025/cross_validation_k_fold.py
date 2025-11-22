from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Initialize Decision Tree classifier
clf = DecisionTreeClassifier(random_state=42)

# Set up K-Fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation
scores = cross_val_score(clf, X, y, cv=kf)

# Print results
print("K-Fold cross-validation scores:", scores)
print("Mean accuracy:", scores.mean())
