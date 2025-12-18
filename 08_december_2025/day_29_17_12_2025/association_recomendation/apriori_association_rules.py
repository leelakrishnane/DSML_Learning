import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

df = pd.read_csv("fashion_boutique_dataset.csv")

print(df.head())
print(df.columns)

df = df[['purchase_date', 'category']]

# Remove missing values
df.dropna(inplace=True)

# Convert purchase_date to date only (remove time if present)
df['purchase_date'] = pd.to_datetime(df['purchase_date']).dt.date

basket = (
    df
    .groupby(['purchase_date', 'category'])['category']
    .count()
    .unstack()
    .fillna(0)
)

# Convert counts to binary
basket = basket.applymap(lambda x: 1 if x > 0 else 0)

print("Basket shape:", basket.shape)
basket.head()

frequent_itemsets = apriori(
    basket,
    min_support=0.02,   # adjust if dataset is large
    use_colnames=True
)

frequent_itemsets.sort_values(by="support", ascending=False).head()

rules = association_rules(
    frequent_itemsets,
    metric="lift",
    min_threshold=1.0
)

rules.head()

strong_rules = rules[
    (rules['confidence'] >= 0.3) &
    (rules['lift'] > 1.2)
].sort_values(by='lift', ascending=False)

strong_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]

for _, row in strong_rules.iterrows():
    antecedent = ", ".join(row['antecedents'])
    consequent = ", ".join(row['consequents'])

    print(
        f"If customers buy [{antecedent}], "
        f"they also buy [{consequent}] "
        f"(Confidence: {row['confidence']:.2f}, Lift: {row['lift']:.2f})"
    )
