import pandas as pd

df = pd.read_csv('diamonds.csv')
print(df)

columns = df.columns
print(columns)

numeric_column = []
non_numeric_column = []

for col in columns:
    if df[col].dtype == 'O':
        non_numeric_column.append(col)
    else:
        numeric_column.append(col)
print(numeric_column)
print(non_numeric_column)

