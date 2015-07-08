import pandas as pd
import numpy as np
from matplotlib import pylab, mlab, pyplot
plt = pyplot

df_train = pd.read_csv("../df_train.csv")
df_test = pd.read_csv("../df_test.csv")

'''
use only PdDistrict and Hour for prediction, since
they have the highest information gain.
'''
chosen_features = df_train.iloc[:,[5, 12]]
chosen_features
chosen_features_test = df_test.iloc[:,[5, 12]]

Y = df_train.iloc[:,[2]]


'''
replace categorical values with 0s and 1s, so sklearn
could work with it.
'''
tmp_hour = pd.get_dummies(chosen_features['Hour'])
tmp_dist = pd.get_dummies(chosen_features['PdDistrict'])
frames= [tmp_hour.T,tmp_dist.T]
X = pd.concat(frames)
X = X.T

'''
Do the same for the test data
'''
tmp_hour_test = pd.get_dummies(chosen_features_test['Hour'])
tmp_dist_test = pd.get_dummies(chosen_features_test['PdDistrict'])
frames_test = [tmp_hour_test.T,tmp_dist_test.T]
X_test = pd.concat(frames_test)
X_test = X_test.T

mapping = {k: v for v, k in enumerate(Y.Category.unique())}
[Y.Category.replace(category, mapping[category], inplace=True) for category in mapping]
Y

'''
Get the y values of the test data
'''
y_test = df_test.iloc[:,[2]]
# Dangerous! this step should be done before the separation of the data sets:
mapping = {k: v for v, k in enumerate(y_test.Category.unique())} 
[y_test.Category.replace(category, mapping[category], inplace=True) for category in mapping]
y_test

# Train with decision tree algorithm
from sklearn import tree
decision_tree = tree.DecisionTreeClassifier()
decision_tree  = decision_tree.fit(X, Y)

# Test with the X values of the testing data to predict the Y values
prediction = decision_tree.predict(X_test)
 ''' 
TODO check import
from sklearn.metrics import zero_one_score
y_pred = svm.predict(test_samples)
accuracy = zero_one_score(y_test, prediction)
error_rate = 1 - accuracy
'''

from sklearn.metrics import accuracy_score
accuracy_score(y_test, prediction) # findout what if this is the succes or error rate :)

# TODO vlaidate!!! 




