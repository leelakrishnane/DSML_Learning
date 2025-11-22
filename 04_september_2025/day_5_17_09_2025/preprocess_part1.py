
# import pandas as pd

# df=pd.read_csv('diamonds.csv')
# print(df)

# null_check = df.isnull()
# print(null_check)

# print('***********')

# finding_null = df.isnull().sum()
# print(finding_null)

# print('*************')

# total_null = df.isnull().sum().sum()
# print(total_null)

# print('*************')

# df.dropna(inplace=True)

# """Below code is for filling the null value"""
# null_check = df.isnull()
# print(null_check)

# print('***********')

#below code is to fill the generic values
#df.fillna(1000, inplace=True)

"""below code is to fill specific value (forward filling- if a row exist earlier to this row that value will be filled in below row)"""
#df.ffill(inplace=True)

"""below code is to fill specific value (backward filling- if a row exist below to this row that value will be filled in above row)"""
#df.bfill(inplace=True)

"""Filling the empty cells with specific data on certain columns"""
#df['carat'] = df['carat'].fillna(0.25)

"""Filling the empty cells with specific data on multiple columns with calculation"""
#df[['carat','cut']] = df[['carat','cut']].fillna(df['price'].mean())


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






