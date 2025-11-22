
# import ssl
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV


# Disable SSL certificate verification
# ssl._create_default_https_context = ssl._create_unverified_context
df = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/teleCust1000t.csv')
print(df)

y = df['custcat'].values
y[0:5]

X = df[['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire',
        'gender', 'reside']] .values  #.astype(float)

# X = StandardScaler().fit(X).transform(X.astype(float))

std_obj = StandardScaler()
X = std_obj.fit_transform(X)
X = X.astype(float)
print(X[0:5])


X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)


knn = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
y_pred = knn.predict(X_test)

print("Train Accuracy:", metrics.accuracy_score(y_train, knn.predict(X_train)))
print("Test Accuracy :", metrics.accuracy_score(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)


params = {
    'n_neighbors': range(1, 21),
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

grid = GridSearchCV(KNeighborsClassifier(), params, cv=5)
grid.fit(X_train, y_train)

print("Best parameters:", grid.best_params_)
print("Best score:", grid.best_score_)