import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Read your data
df = pd.read_csv("diamonds.csv")

# Define the columns you want to encode
categorical_columns = ['color', 'cut', 'clarity']

# Create the ColumnTransformer
ct = ColumnTransformer(
    transformers=[
        ('encoder', OneHotEncoder(sparse_output=False, drop=None), categorical_columns)
    ],
    remainder='passthrough'  # Keep all other columns as they are
)

# Fit and transform the data
transformed_data = ct.fit_transform(df)

# Get the encoded column names
encoded_column_names = ct.named_transformers_['encoder'].get_feature_names_out(categorical_columns)

# Build a new DataFrame with the transformed data
final_df = pd.DataFrame(transformed_data, columns=list(encoded_column_names) + [col for col in df.columns if col not in categorical_columns])

# Optional: ensure numeric types are handled properly
final_df = final_df.infer_objects()

print(final_df.head())
