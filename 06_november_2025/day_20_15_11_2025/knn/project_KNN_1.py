import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix

# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------
df = pd.read_csv(
    'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/'
    'IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/teleCust1000t.csv'
)

# --------------------------------------------------
# 2. SPLIT FEATURES AND TARGET
# --------------------------------------------------
y = df['custcat'].values
X = df[['region', 'tenure','age', 'marital', 'address', 'income',
        'ed', 'employ','retire','gender', 'reside']].values

# --------------------------------------------------
# 3. STANDARDIZE FEATURES
# --------------------------------------------------
scaler = StandardScaler()
X = scaler.fit_transform(X).astype(float)

# --------------------------------------------------
# 4. TRAIN / TEST SPLIT
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=4
)

print('Train set:', X_train.shape,  y_train.shape)
print('Test set:', X_test.shape,  y_test.shape)

# --------------------------------------------------
# 5. BASE KNN MODEL (k = 5)
# --------------------------------------------------
k = 5
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print("\n=== BASIC KNN (k=4) ===")
print("Train Accuracy:", metrics.accuracy_score(y_train, knn.predict(X_train)))
print("Test Accuracy :", metrics.accuracy_score(y_test, y_pred))

# --------------------------------------------------
# 6. FIND BEST k VALUE (1–15)
# --------------------------------------------------
Ks = 15
accuracy_list = np.zeros(Ks)

for n in range(1, Ks+1):
    model = KNeighborsClassifier(n_neighbors=n)
    model.fit(X_train, y_train)
    y_pred_test = model.predict(X_test)
    accuracy_list[n-1] = metrics.accuracy_score(y_test, y_pred_test)

best_k = accuracy_list.argmax() + 1
print("\n=== OPTIMIZATION RESULTS ===")
print("Accuracy for each k:", accuracy_list)
print("Best accuracy:", accuracy_list.max())
print("Best k:", best_k)

# Plot (optional)
plt.figure(figsize=(8,4))
plt.plot(range(1, Ks+1), accuracy_list, 'bo-', linewidth=2)
plt.xlabel('k value')
plt.ylabel('Accuracy')
plt.title('Accuracy vs k')
plt.grid(True)
plt.show()

# --------------------------------------------------
# 7. FINAL MODEL USING BEST k
# --------------------------------------------------
final_knn = KNeighborsClassifier(n_neighbors=best_k)
final_knn.fit(X_train, y_train)
y_final = final_knn.predict(X_test)
print("\n=== FINAL MODEL ===")
print("Final Test Accuracy:", metrics.accuracy_score(y_test, y_final))

# --------------------------------------------------
# 8. CONFUSION MATRIX
# --------------------------------------------------
cm = confusion_matrix(y_test, y_final)
print("\nConfusion Matrix:\n", cm)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
