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
frames_test = [tmp_hour.T,tmp_dist.T]
X_test = pd.concat(frames_test)
X_test = X_test.T

mapping = {k: v for v, k in enumerate(Y.Category.unique())}
[Y.Category.replace(cat, mapping[cat], inplace=True) for cat in mapping]
Y

# Train with decision tree algorithm
from sklearn import tree
decision_tree = tree.DecisionTreeClassifier()
decision_tree  = decision_tree.fit(X, Y)

# Test with the X values of the testing data to predict the Y values
prediction = decision_tree.predict(X_test)
# TODO vlaidate!!! 




