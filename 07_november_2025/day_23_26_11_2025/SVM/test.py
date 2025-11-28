from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

data = load_breast_cancer()
print(data.feature_names)
idx = np.random.randint(0, data.data.shape[0])
print("Random index:", idx)
print("Random row data:", data.data[idx])