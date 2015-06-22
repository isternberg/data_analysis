import pandas as pd
import numpy as np
from matplotlib import pylab, mlab, pyplot
plt = pyplot

df = pd.read_csv("../crime_train.csv")

def getYear(s):
  return s.split("-")[0]
def getMonth(s):
  return s.split("-")[1]

df['year']= df['Dates'].apply(lambda x: getYear(x))
df['month']= df['Dates'].apply(lambda x: getMonth(x))
ct = pd.crosstab(df.Category, df.year)
a = ct.ix[23:24].T
a
a.describe()
a.plot()

# something to work on:
'''
pd.to_datetime(df['Dates'])

dt = df['Dates'].to_frame()
a = pd.to_datetime(df['Dates'])
dt = a.to_frame()
'''

