import pandas as pd

data = pd.read_csv("diamonds.csv")
print(data)

print("-----------------------")

# """below two lines prints the column names of the csv file"""
# column_names = data.columns
# print(column_names)
#
# print("***************************************")
#
# """below two lines describes about the dataset"""
# data_description = data.describe()
# print(data_description)
#
# print("*****************************************")
#
# """below codes shows the data types of each of the columns"""
# column_data_types =  data.dtypes
# print(column_data_types)

# data_information = data.info()
# print(data_information)

"""below code is to return the rows of the given one column"""
# data_column = data['cut']
# print(data_column)

"""below syntax is to return rows of more than one column"""
# data_column = data[['cut',"carat","x"]]
# print(data_column)

# data_rows_top = data.head()
# print(data_rows_top)

# data_rows_bottom = data.tail()
# print(data_rows_bottom)
#
# data_rows_top_specification = data.head(10)
# print(data_rows_top_specification)
#
# data_rows_down_specification = data.tail(10)
# print(data_rows_down_specification)

"""below syntax of retrieving rows with index number is called slicing"""
# data_from_range = data[0:10] #0th index to 9th index
# print(data_from_range)
#
# # data_from_range = data[:20] #0th to 19th index
# # print(data_from_range)
#
# data_from_range = data[10:] #10th to last index
# print(data_from_range)

"""below three is similar to slicing but has one difference which in the last element"""

# data_from_range = data.loc[0:10] #0th index to 10th index
# print(data_from_range)
#
# # data_from_range = data.loc[:20] #0th to 20th index
# # print(data_from_range)
#
# data_from_range = data.loc[10:] #10th to last index
# print(data_from_range)

"""below three lines is to fetch rows with columns"""

data_from_range = data.loc[0:10,"carat"] #0th index to 10th index
print(data_from_range)

data_from_range = data.loc[:20,["carat","cut","x"]] #0th to 20th index
print(data_from_range)

# data_from_range = data.loc[10:] #10th to last index
# print(data_from_range)