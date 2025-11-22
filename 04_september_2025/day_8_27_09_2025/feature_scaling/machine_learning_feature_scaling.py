import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

df = pd.read_csv('diamonds.csv')
print(df)

# standar_scaler_obj = StandardScaler()
# df["price"] = standar_scaler_obj.fit_transform(df[["price"]])
# print(df)


min_max_scaler_obj = MinMaxScaler()
df["price"] = min_max_scaler_obj.fit_transform(df[["price"]])
print(df)