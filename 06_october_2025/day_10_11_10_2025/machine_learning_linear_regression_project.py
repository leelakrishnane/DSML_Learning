import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

df = pd.read_csv('Ecommerce_Customers.csv')
print(df)


sns.pairplot(df)
plt.show()

X = df[['Avg. Session Length','Time on App','Time on Website','Length of Membership']]

y = df['Yearly Amount Spent']


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = LinearRegression()
model.fit(x_train, y_train)
final_prediction = model.predict(x_test)
print("***************************")
print(final_prediction)
print("***************************")

metric_evaluation = r2_score(y_test, final_prediction)
print("R2 Score:", metric_evaluation)

