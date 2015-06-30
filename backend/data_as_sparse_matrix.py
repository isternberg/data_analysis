import pandas as pd
import numpy as np
from matplotlib import pylab, mlab, pyplot
plt = pyplot

df = pd.read_csv("../crime_train.csv")
# split the Dates column to Year and Month in order to remove
# the data for 2015, which is incomplete
def getYear(s):
  return s.split("-")[0]
def getMonth(s):
  return s.split("-")[1]

df['Year']= df['Dates'].apply(lambda x: getYear(x))
df['Month']= df['Dates'].apply(lambda x: getMonth(x))
df['Year'] = df['Year'].apply(int)
df['Month'] = df['Month'].apply(int)
# remove the data for the year 2015
df = df[df.Year != 2015]
# confirm 2015 was really removed
years = df.Year.unique()
years

tmp_days = pd.get_dummies(df['DayOfWeek'])
tmp_cat =  pd.get_dummies(df['Category'])
tmp_dist = pd.get_dummies(df['PdDistrict'])
frames= [tmp_days.T, tmp_cat.T, tmp_dist.T]
sparse_df = pd.concat(frames)
sparse_df = sparse_df.T

# shufle the data
sparse_df = sparse_df.reindex(np.random.permutation(df.index))
# keep 80% of the data for training. The other 20% will be testing data
training_len = int(len(sparse_df)* 0.8)
testing_len = len(sparse_df) - training_len
df_train = sparse_df.head(training_len)
df_test = sparse_df.tail(testing_len)
#df_train.to_csv("../sparse_df_train.csv",  encoding='utf-8')
#df_test.to_csv("../sparse_df_test.csv",  encoding='utf-8')

