import pandas as pd
import numpy as np
from matplotlib import pylab, mlab, pyplot
plt = pyplot

df = pd.read_csv("../crime_train.csv")
def getYear(s):
  return s.split("-")[0]
def getMonth(s):
  return s.split("-")[1]
def getHour(s):
  tmp = s.split(" ")[1]
  return tmp.split(":")[0]

df['Year']= df['Dates'].apply(lambda x: getYear(x))
df['Month']= df['Dates'].apply(lambda x: getMonth(x))
df['Hour']= df['Dates'].apply(lambda x: getHour(x))
df['Year'] = df['Year'].apply(int)
df['Month'] = df['Month'].apply(int)
df['Hour'] = df['Hour'].apply(int)
# remove the data for the year 2015
df = df[df.Year != 2015]
#test is the yeaar 2015 was really removed
years = df.Year.unique()
years

# shufle the data
df = df.reindex(np.random.permutation(df.index))
# keep 80% of the data for training. The other 20% will be testing data
training_len = int(len(df)* 0.8)
testing_len = len(df) -training_len
df_train = df.head(training_len)
df_test = df.tail(testing_len)

training_data = df_train.iloc[:,[4, 11, 1]]
