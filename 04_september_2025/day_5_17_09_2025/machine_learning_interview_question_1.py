import pandas as pd

df=pd.read_csv('diamonds.csv')
print(df)

column = df.columns
print(column)

numeric_column = []
non_numeric_column = []

for col in column:
    if df[col].dtype == 'O':
        non_numeric_column.append(col)
    else:
        numeric_column.append(col)

print(numeric_column)
print(non_numeric_column)


# paranthesis () means tuple
# Square bracket means 