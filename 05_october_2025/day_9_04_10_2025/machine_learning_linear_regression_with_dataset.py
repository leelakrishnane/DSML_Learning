import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("diamonds.csv")
print(df)

finding_null = df.isnull().sum()
print(finding_null)

df.fillna(method='ffill', inplace=True)


column_names = df.columns
print(column_names)
print("-------------------------")

df['cut'] = df['cut'].map({"Ideal":1,"Fair":2,"Good":3,"Very Good":4,"Premium":5})
print(df)

print("-------------------------")


df['clarity'] = df['clarity'].map({"I1":1,"SI2":2,"SI1":3,"VS2":4,"VS1":5,"VVS2":6,"VVS1":7,"IF":8})
print(df)

# label_encoder_object = LabelEncoder()
# df['cut'] = label_encoder_object.fit_transform(df['cut'])
# print(df['cut'])

print("-------------------------")

x = df[["carat","cut","depth","table","x","y","z"]]
y= df["price"]



model_obj = LinearRegression()
model_obj.fit(x,y)


output_df = pd.DataFrame({"carat":[0.3,0.2,0.33],"cut":[2,1,2],"depth":[61.5,59.8,62.1],"table":[55,57,54],"x":[3.5,3.0,3.2],"y":[3.5,3.0,3.2],"z":[2.5,2.0,2.2]})
print(output_df)
print("-------------------------")

final_prediction = model_obj.predict(output_df)
print(final_prediction)
print("-------------------------")

output_df['predicted_price'] = final_prediction
print(output_df)