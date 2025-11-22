from sklearn.linear_model import LinearRegression
import pandas as pd

data = {"year": [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010],
        "House_Price": [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000]}

df = pd.DataFrame(data)
print(df)
print("-------------------------")
X = df[["year"]]
y = df["House_Price"]

model_obj = LinearRegression()
model_obj.fit(X, y)

output_df = pd.DataFrame({"year":[2007,2011,2015]})
print(output_df)
print("-------------------------")

final_prediction = model_obj.predict(output_df)
print(final_prediction)
print("-------------------------")

output_df['predicted_price'] = final_prediction
print(output_df)