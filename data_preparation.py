import pandas as pd
import numpy as np
from sklearn import cross_validation
import numpy.testing as npt

df = pd.read_csv("../crime_train.csv")

# 3.1 data preprocessing
# split the Dates column into Year, Month, Hour
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

# remove the data for the year 2015, which is incomplete
df = df[df.Year != 2015]
# test is the year 2015 was really removed
years = df.Year.unique()
# test that 2015 is no longer part of the dataset
npt.assert_equal(years.max(), 2014)

# every Category get's an ID
Cat_num = df.Category.copy(deep=True)
mapping = {k: v for v, k in enumerate(Cat_num.unique())}
[Cat_num.replace(category, mapping[category] , inplace=True) for category in mapping]
df["Cat_num"] = Cat_num
# test if the number of unique categories is the same as unique category numbers
npt.assert_equal(len(df.Cat_num.unique()), len(df.Category.unique()))

# shuffle the data
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

# 3.1.3 data preparation
#remove the crimes, which have less than 100 instances in the training data
tmp =df_train.groupby("Category").count().sort_index(by=['Year'], ascending=[True])
tmp = tmp.iloc[:,[0]]
print(tmp)
df_train = df_train[df_train.Category != "TREA"]
df_train = df_train[df_train.Category != "PORNOGRAPHY/OBSCENE MAT"]
df_test = df_test[df_test.Category != "TREA"]
df_test = df_test[df_test.Category != "PORNOGRAPHY/OBSCENE MAT"]

# 3.2 Determination of relevant features
# keep only the columns that are interesting for the prediction
# ['DayOfWeek', 'PdDistrict', 'Month', 'Hour', 'Cat_num']
def reduce_to_relevant_columns(dataframe):
  df_reduced = dataframe.iloc[:,[3,4,10,11,12]]
  return df_reduced

df_train_reduced = reduce_to_relevant_columns(df_train)
df_test_reduced = reduce_to_relevant_columns(df_test)

# test that the desired columns were kept
features = ['DayOfWeek', 'PdDistrict', 'Month', 'Hour', 'Cat_num']
npt.assert_array_equal(features, df_train_reduced.columns)
npt.assert_array_equal(features, df_test_reduced.columns)


#number_of_features = 2    # features: 'DayOfWeek', 'PdDistrict'
number_of_features = 3    # features: 'DayOfWeek', 'PdDistrict', 'Month'
#number_of_features = 4    # features: 'DayOfWeek', 'PdDistrict', 'Month', 'Hour'
'''
replace categorical values of features with 0s and 1.
'''
def create_dummies(dataFrame, number_of_features):
  tmp_dist = pd.get_dummies(dataFrame['PdDistrict'])
  tmp_hour = pd.get_dummies(dataFrame['Hour'])
  tmp_day = pd.get_dummies(dataFrame['DayOfWeek'])
  tmp_month = pd.get_dummies(dataFrame['Month'])
  frames= [tmp_dist.T, tmp_hour.T, tmp_day.T, tmp_month.T]
  new_df = pd.concat(frames[0:number_of_features])
  new_df = new_df.T
  # Add The Category-number column
  new_df["Category"] = dataFrame["Cat_num"]
  return new_df

df_train_reduced = create_dummies(df_train_reduced, number_of_features)
df_test_reduced = create_dummies(df_test_reduced, number_of_features)

features[0] = ['BAYVIEW','CENTRAL','INGLESIDE','MISSION','NORTHERN','PARK','RICHMOND',
        'SOUTHERN','TARAVAL','TENDERLOIN']
features[1]= [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
features[2] = ['Friday','Monday','Saturday','Sunday','Thursday','Tuesday','Wednesday']
features[3] = [1,2,3,4,5,6,7,8, 9,10,11,12]
category = ['Category']

# test the desired columns are there after creating the dummies
def create_cols(number_of_features):
    cols = []
    for i in range(number_of_features):
        cols += features[i]
    cols.extend(category)
    return cols

expected_cols = create_cols(number_of_features)
npt.assert_array_equal(list(df_train_reduced.columns.values),expected_cols)

column_range = df_train_reduced.columns.size - 1
# devide the data to X (features) and Y (value for prediction)
y_train = df_train_reduced.Category
x_train = df_train_reduced.drop('Category', axis=1)
#list(x_train.columns.values)
# Test the separation to X and Y was done correctly
expected_x_cols = expected_cols[:column_range]
expected_y_cols = expected_cols[column_range:column_range+1]
npt.assert_array_equal(list(x_train.columns.values), expected_x_cols)
npt.assert_array_equal(list(y_train.to_frame().columns.values), expected_y_cols)

y_test = df_test_reduced.Category
x_test = df_test_reduced.drop('Category', axis=1)
# list(x_test.columns.values)
# Test the separation to X and Y was done correctly
npt.assert_array_equal(list(x_test.columns.values), expected_x_cols)
npt.assert_array_equal(list(y_test.to_frame().columns.values), expected_y_cols)




