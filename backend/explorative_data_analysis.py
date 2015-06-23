
import pandas as pd
import numpy as np
from matplotlib import pylab, mlab, pyplot
plt = pyplot

df = pd.read_csv("../crime_train.csv")

ct = pd.crosstab(df.Category, df.DayOfWeek)
display = ct.ix[:,['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']] 
print(display)
prostitution_row = display.ix[23:24] 
print(prostitution_row)
print(prostitution_row.T)
print(prostitution_row.T.describe)

display.head()
prostitution_row
prostitution_row.T.describe()

pgroup=df['Category'].groupby(df['PdDistrict'])
a = pgroup.count().to_frame()
a.columns=["Sum_of_crimes"]




ct = pd.crosstab(df.DayOfWeek, df.Category)
ct.values
tmp = ct.values

# Following 4 functions are taken from 
# http://christianherta.de/lehre/dataScience/machineLearning/decision-trees.php
def p_log_p(p_):
  p = p_.copy()
  p[p != 0] = - p[p != 0] * np.log2(p[p != 0] )
  return p

def entropy(p):
  assert (p>=0.).all()
  assert np.allclose(1., p.sum(), atol=1e-08)
  return p_log_p(p).sum()
  
def conditional_entropy(p, axis=0):
  if axis == 1:
    p = p.T
  p_y_given_x = p / p.sum(axis=0)
  c = p_log_p(p_y_given_x) 
  p_x = p.sum(axis=0)/p.sum()
  s = c * p_x
  return s.sum()
 
def information_gain(p0, axis = 0):
  p = p0.copy()
  if axis == 1:
    p = p.T
  p_ = p.sum(axis=1)
  return entropy(p_) - conditional_entropy(p)

p = tmp * 1.0 /tmp.sum()
p_ = p.sum(axis=1)
entropy(p_)
conditional_entropy(p)
information_gain(p)

gain = information_gain(p) / entropy(p_)
print(gain)
