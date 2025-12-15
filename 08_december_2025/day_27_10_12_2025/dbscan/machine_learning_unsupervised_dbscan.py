from sklearn.datasets import load_iris
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


iris = load_iris()
x = iris.data

x_scaled = StandardScaler().fit_transform(x)

model = DBSCAN(eps=0.65, min_samples=4)
output = model.fit_predict(x_scaled)

print()
