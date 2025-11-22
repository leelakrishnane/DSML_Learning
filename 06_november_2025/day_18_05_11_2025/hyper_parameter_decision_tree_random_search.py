from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define the model
my_model = DecisionTreeClassifier(random_state=42)

# Define hyperparameters to tune
my_param = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 3, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Set up RandomizedSearchCV
my_rand = RandomizedSearchCV(estimator=my_model, param_distributions=my_param,
                           cv=5, n_jobs=-1, scoring='accuracy')

# Fit the model
my_rand.fit(X_train, y_train)

# Best parameters and score
print("Best Parameters:", my_rand.best_params_)
print("Best Cross-Validation Score:", my_rand.best_score_)

# Evaluate on test set
best_model = my_rand.best_estimator_
y_pred = best_model.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_pred))