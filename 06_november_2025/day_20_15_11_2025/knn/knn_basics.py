from sklearn.neighbors import KNeighborsRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

boston = fetch_california_housing()

x=boston.data
y=boston.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = KNeighborsRegressor(n_neighbors=3)
model.fit(x_train, y_train)
predictions = model.predict(x_test) 
print(predictions)


model_evaluation = r2_score(y_test, predictions)
print("R2 Score:", model_evaluation)

