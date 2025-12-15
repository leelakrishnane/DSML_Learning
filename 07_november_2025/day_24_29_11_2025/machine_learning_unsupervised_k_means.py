import pandas as pd
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

data = load_iris()

X = data.data
y = data.target
print(X.shape)

model = KMeans(n_clusters=3, random_state=42)
model.fit_predict(X)

label = model.labels_
print(label)

plt.figure(figsize=(10,8))
plt.show()