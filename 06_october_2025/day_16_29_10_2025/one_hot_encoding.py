import pandas as pd

df = pd.read_csv('diamonds.csv')

print (df)

converted_df = pd.get_dummies(df, columns=['color'], prefix=['color'])
print(converted_df.head())

df.drop(columns=['color'], axis=1, inplace=True)

final_df = pd.concat([df, converted_df], axis=1)
print(final_df.columns())