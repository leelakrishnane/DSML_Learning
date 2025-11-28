from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_iris

iris = load_iris()

X,y = iris.data, iris.target

knn = 