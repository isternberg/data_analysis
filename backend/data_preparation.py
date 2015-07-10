import pandas as pd
import numpy as np
from sklearn import cross_validation
import numpy.testing as npt

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
# test that 2015 is no longer there
npt.assert_equal(years.max(), 2014)


Cat_num = df.Category.copy(deep=True)
mapping = {k: v for v, k in enumerate(Cat_num.unique())}
[Cat_num.replace(category, mapping[category] , inplace=True) for category in mapping]
df["Cat_num"] = Cat_num
# test if the number of unique categories is the same as unique category numbers
npt.assert_equal(len(df.Cat_num.unique()), len(df.Category.unique())) 

# shufle the data
df = df.reindex(np.random.permutation(df.index))
# keep 80% of the data for training. The other 20% will be testing data
training_len = int(len(df)* 0.8)
testing_len = len(df) - training_len
df_train = df.head(training_len)
df_test = df.tail(testing_len)
# test that now rows were added or removed
npt.assert_equal(len(df), len(df_train) + len(df_test))
# save the new training data as a file
df_train.to_csv("../df_train.csv",  encoding='utf-8')



# keep only the columns that are interesting for the prediction
def reduce_to_relevant_columns(dataframe):
  df_reduced = dataframe.iloc[:,[3,4,10,11,12]]
  return df_reduced
  
df_train_reduced = reduce_to_relevant_columns(df_train)
df_test_reduced = reduce_to_relevant_columns(df_test)

# test that the desired columns were kept
cols = ['DayOfWeek', 'PdDistrict', 'Month', 'Hour', 'Cat_num']
npt.assert_array_equal(cols, df_train_reduced.columns)
npt.assert_array_equal(cols, df_test_reduced.columns)

'''
replace categorical values of features with 0s and 1.
'''
def create_dummies(dataFrame):
  tmp_hour = pd.get_dummies(dataFrame['Hour'])
  tmp_dist = pd.get_dummies(dataFrame['PdDistrict'])
  tmp_day = pd.get_dummies(dataFrame['DayOfWeek'])
  tmp_month = pd.get_dummies(dataFrame['Month'])
  frames= [tmp_hour.T,tmp_dist.T, tmp_day.T, tmp_month.T]
  new_df = pd.concat(frames)
  new_df = new_df.T
  # Add The Category-number column
  new_df["Category"] = dataFrame["Cat_num"]
  return new_df

df_train_reduced = create_dummies(df_train_reduced)
df_test_reduced = create_dummies(df_test_reduced)

# test the desired columns are there after creating the dummies
expected_cols = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,
        'BAYVIEW','CENTRAL','INGLESIDE','MISSION','NORTHERN','PARK','RICHMOND',
        'SOUTHERN','TARAVAL','TENDERLOIN',
        'Friday','Monday','Saturday','Sunday','Thursday','Tuesday','Wednesday',
        1,2,3,4,5,6,7,8, 9,10,11,12,'Category']
npt.assert_array_equal(list(df_train_reduced.columns.values),expected_cols)



y_train = df_train_reduced.Category
x_train = df_train_reduced.drop('Category', 1)
list(x_train.columns.values)
# Test the spearation to X and Y was done correctly
expected_x_cols = expected_cols[:53]
expected_y_cols = expected_cols[53:54]
npt.assert_array_equal(list(x_train.columns.values), expected_x_cols)
npt.assert_array_equal(list(y_train.to_frame().columns.values), expected_y_cols)

y_test = df_test_reduced.Category
x_test = df_test_reduced.drop('Category', 1)
list(x_test.columns.values)
# Test the spearation to X and Y was done correctly
npt.assert_array_equal(list(x_test.columns.values), expected_x_cols)
npt.assert_array_equal(list(y_test.to_frame().columns.values), expected_y_cols)


# Train with decision tree algorithm
from sklearn import tree
decision_tree = tree.DecisionTreeClassifier()

#cv is number of folds
scores_tree = cross_validation.cross_val_score(decision_tree, x_train, y_train, cv=5)

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
score_bayes = cross_validation.cross_val_score(gnb, x_train, y_train, cv=5)


from sklearn.neighbors.nearest_centroid import NearestCentroid
clf = NearestCentroid()
score_nearest_centroid = cross_validation.cross_val_score(clf, x_train, y_train, cv=5)




# train
decision_tree  = decision_tree.fit(x_train, tmp.y_train)
# predict - TODO the same with gnb and clf
prediction = decision_tree.predict(x_test)
# test
from sklearn.metrics import accuracy_score
accuracy_score(y_test, prediction)  # 0.2





