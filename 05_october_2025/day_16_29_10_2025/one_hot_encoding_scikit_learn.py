import pandas as pd
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv('diamonds.csv')

one_hot_encoding = OneHotEncoder(sparse_output=False)  
encoded_data = one_hot_encoding.fit_transform(df[['color']])

print(encoded_data)

encoded_column_name = one_hot_encoding.get_feature_names_out()
print(encoded_column_name)

encoded_df = pd.DataFrame(encoded_data, columns=encoded_column_name)
print(encoded_df)

final_df = pd.concat([df, encoded_df], axis=1)
print(final_df)