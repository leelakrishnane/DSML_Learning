# import pandas as pd
#
# df = pd.read_csv('diamonds.csv')
# print(df)
#
# finding_null = df.isnull().sum()
# print(finding_null)
# print("***********************")
# total_null = df.isnull().sum().sum()
# print(total_null)
#
# df.dropna(inplace=True)
#
# finding_null = df.isnull().sum()
# print(finding_null)
# print("***********************")
# total_null = df.isnull().sum().sum()
# print(total_null)

"""below code is for filling the null value"""
# import pandas as pd
#
# df = pd.read_csv('diamonds.csv')
# print(df)
#
# finding_null = df.isnull().sum()
# print(finding_null)
# print("***********************")
#
# """below code is to fill a generic value"""
# #df.fillna(1000, inplace=True)
#
#
# """below code is to fill a specific value(forward filling - if a row exist earlier to this row that value will be filled in below row)"""
# #df.ffill(inplace=True)
#
# """below code is to fill a specific value(backward filling - if a row exist below to this row that value will be filled in above row)"""
# #df.bfill(inplace=True)
#
# #df[['carat','cut']] = df[['carat','cut']].fillna(0.25)
#
# df[['carat','cut']] = df[['carat','cut']].fillna(df['price'].mean())
#
# print(df)
# finding_null = df.isnull().sum()
# print(finding_null)
#print("***********************")

"""below code is to filling null data with new library"""
import pandas as pd
from sklearn.impute import SimpleImputer

df = pd.read_csv('diamonds.csv')
print(df)

finding_null = df.isnull().sum()
print(finding_null)
print("***********************")

simple_imputer_object = SimpleImputer()
df['price'] = simple_imputer_object.fit_transform(df[['price']])

print(df)
finding_null = df.isnull().sum()
print(finding_null)
