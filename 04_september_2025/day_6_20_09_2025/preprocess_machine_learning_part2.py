import pandas as pd
df = pd.read_csv('diamonds.csv')
print(df)

# finding_category_count = df['cut'].value_counts()
# print(finding_category_count)

# finding_category_count_total = df['cut'].value_counts().sum()
# print(finding_category_count_total)

# unique_category_names = df['cut'].unique()
# print(unique_category_names)

# total_unique_category_names = df['cut'].nunique()
# print(total_unique_category_names)

# df['cut'] = df['cut'].map({"Ideal":0,"Fair":1,"Good":2,"Very Good":3,"Premium":4})
# print(df)

from sklearn.preprocessing import LabelEncoder
print(df['color'])
print("********")
label_encoder_object = LabelEncoder()
df['color'] = label_encoder_object.fit_transform(df['color'])
print(df['color'])


print(label_encoder_object.classes_)

print(label_encoder_object.transform(label_encoder_object.classes_))

class_label = label_encoder_object.classes_
class_label_number = label_encoder_object.transform(label_encoder_object.classes_)
testing_mapping = dict(zip(class_label, class_label_number))
print(testing_mapping)


# from sklearn.preprocessing import OrdinalEncoder
# print(df['color'])
# print("********")
# ordinal_encoder_object = OrdinalEncoder()
# df['color'] = ordinal_encoder_object.fit_transform(df[['color']])
# print(df['color'])

"""Ordinal encoder will not transform a NaN/Empty value, but the label encoder will even consider the NaN/Empty values also"""