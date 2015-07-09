import pandas as pd
import numpy as np
from matplotlib import pylab, mlab, pyplot
plt = pyplot

df = pd.read_csv("../crime_train.csv")
# split the Dates column in order to remove
# the data for 2015, which is incomplete
def getYear(s):
  return s.split("-")[0]
def getMonth(s):
  return s.split("-")[1]
def getHour(s):
  tmp = s.split(" ")[1]
  return tmp.split(":")[0]

df['Year']= df['Dates'].apply(lambda x: getYear(x)).apply(int)
df['Month']= df['Dates'].apply(lambda x: getMonth(x)).apply(int)
df['Hour']= df['Dates'].apply(lambda x: getHour(x)).apply(int)

# remove the data for the year 2015
df = df[df.Year != 2015]
#test is the yeaar 2015 was really removed
years = df.Year.unique()
years
# TODO assert

# keep only the columns that are interessting for the prediction
df_reduced = df.iloc[:,[3,4,10,11,1]]
# TODO assert


# before manipulation the data, make a copy of it
# df_for_prediction = df.copy(deep=True)

mapping = {k: v for v, k in enumerate(df_reduced.Category.unique())}
[df_reduced.Category.replace(category, mapping[category], inplace=True) for category in mapping]
df_reduced.head(5)
df.head(5)

'''
replace categorical values of features with 0s and 1s, so sklearn
could work with it.
'''
tmp_hour = pd.get_dummies(df_reduced['Hour'])
tmp_dist = pd.get_dummies(df_reduced['PdDistrict'])
tmp_day = pd.get_dummies(df_reduced['DayOfWeek'])
tmp_month = pd.get_dummies(df_reduced['Month'])
frames= [tmp_hour.T,tmp_dist.T, tmp_day.T, tmp_month.T]
df_reduced_numeric = pd.concat(frames)
df_reduced_numeric = df_reduced_numeric.T
# Add The Category column
df_reduced_numeric["Category"] = df_reduced.Category
# list all column names to see we got the right structure
list(df_reduced_numeric.columns.values)
# TODO assert column names

# shufle the data
df_reduced_numeric = df_reduced_numeric.reindex(np.random.permutation(df_reduced_numeric.index))
# keep 80% of the data for training. The other 20% will be testing data
training_len = int(len(df_reduced_numeric)* 0.8)
testing_len = len(df_reduced_numeric) -training_len
# TODO assert len(df_reduced_numeric) - training_len = testing_len

df_train = df_reduced_numeric.head(training_len)
df_test = df_reduced_numeric.tail(testing_len)
# TODO assert no row skippped
y_train = df_train.Category
x_train = df_train.drop('Category', 1)
list(x_train.columns.values)
# TODO assert column names

y_test = df_test.Category
x_test = df_test.drop('Category', 1)
list(x_test.columns.values)
# TODO assert column names



# Train with decision tree algorithm
from sklearn import tree
decision_tree = tree.DecisionTreeClassifier()
# train
decision_tree  = decision_tree.fit(x_train, y_train)
# predict
prediction = decision_tree.predict(x_test)
# test
from sklearn.metrics import accuracy_score
accuracy_score(y_test, prediction)  # 0.2




