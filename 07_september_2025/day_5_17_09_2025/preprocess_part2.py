from sklearn.impute import SimpleImputer
import pandas as pd

df=pd.read_csv('diamonds.csv')
print(df)

simple_imputer_object = SimpleImputer()
df['price'] = simple_imputer_object.fit_transform(df[['price']])

print(df)

print('*************')

finding_null = df.isnull().sum()
print(finding_null)
