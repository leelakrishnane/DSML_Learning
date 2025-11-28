
"""
Logistic regression is a type of supervised learning algorithm that is used for
classification problems.

It is a statistical method that is used to fit a model to a dataset,
where the goal is to predict a binary outcome
(e.g. a yes/no, true/false, or 0/1 response) based on one or
more predictor variables (also known as features or input variables).

The basic idea behind logistic regression is to use a linear function of the
input variables to model the probability of the binary outcome.

Logistic regression is typically used when the response variable is binary,
although it can also be used for ordinal or multinomial data.

It is a widely used method in many fields,
including medical research, social sciences,
marketing, and finance.

Logistic regression is used to predict the probability of an event occurring,
such as a customer buying a product,
a patient developing a certain disease,
or a voter supporting a particular candidate.

It can also be used to model the relationship between multiple predictor variables and the binary outcome.

When using scikit-learn to perform logistic regression, data must first be loaded and preprocessed before being used to fit a model.
Here's an example of how to load a dataset and preprocess it for logistic regression using scikit-learn:
"""
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# load the iris dataset as an example
my_data = load_iris()
print(my_data)

print(my_data.feature_names)
print(my_data.target_names)

X = my_data.data
y = my_data.target


# # split the data into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
#
#
# """
# we first load the iris dataset using the load_iris function from scikit-learn.
# Then, we split the dataset into training and test sets using the train_test_split function.
# Finally, we standardize the data using the StandardScaler class from scikit-learn.
# This is done by first fitting the scaler to the training data and
# then transforming both the training and test data using the transform method.
# """
# # standardize the data
# scaler = StandardScaler()
# scaler.fit(X_train)
# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)
#
# """
# **3. How to fit a logistic regression model to data using scikit-learn,
#  including how to handle categorical variables
#  and how to interpret the model coefficients.**
#
# """
#
# from sklearn.linear_model import LogisticRegression
#
# # create a LogisticRegression object
# log_reg = LogisticRegression()
#
# # fit the model to the data
# log_reg.fit(X_train, y_train)
#
# # make predictions
# y_pred = log_reg.predict(X_test)
# print(y_pred)
#
# from sklearn.metrics import accuracy_score
#
# final_accuracy_check = accuracy_score(y_test, y_pred)
# print(final_accuracy_check)