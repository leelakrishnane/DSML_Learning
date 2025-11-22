import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv('social_media.csv')
print(df.head())

finding_null = df.isnull().sum()
print(finding_null)

label_encoder_object = LabelEncoder()
df['Gender'] = label_encoder_object.fit_transform(df['Gender'])
df['Academic_Level'] = label_encoder_object.fit_transform(df['Academic_Level'])
df['Country'] = label_encoder_object.fit_transform(df['Country'])
df['Most_Used_Platform'] = label_encoder_object.fit_transform(df['Most_Used_Platform'])
df['Affects_Academic_Performance'] = label_encoder_object.fit_transform(df['Affects_Academic_Performance'])

print(df[['Gender', 'Academic_Level', 'Country', 'Most_Used_Platform', 'Affects_Academic_Performance']])

x = df[["Age","Gender","Academic_Level","Country","Most_Used_Platform","Avg_Daily_Usage_Hours","Affects_Academic_Performance"]]
y= df["Addicted_Score"]

model_obj = LinearRegression()
model_obj.fit(x,y)

output_df = pd.DataFrame({"Age":[20,21,22],"Gender":[1,0,1],"Academic_Level":[2,1,2],"Country":[0,1,0],"Most_Used_Platform":[1,0,1],"Avg_Daily_Usage_Hours":[5,3,4],"Affects_Academic_Performance":[1,0,1]})
print(output_df)
print("-------------------------")

final_prediction = model_obj.predict(output_df)
print(final_prediction)
print("-------------------------")

output_df['predicted_addicted_score'] = final_prediction
print(output_df)