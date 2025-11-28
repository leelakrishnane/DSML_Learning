from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Initialize Decision Tree classifier
clf = DecisionTreeClassifier(random_state=42)

# Set up Stratified K-Fold cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation
scores = cross_val_score(clf, X, y, cv=skf)

# Print results
print("Stratified K-Fold cross-validation scores:", scores)
print("Mean accuracy:", scores.mean())
