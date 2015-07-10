import pandas as pd
import numpy as np


df = pd.read_csv("../df_train.csv")

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

# count of each crime category by DayOfWeek
ct = pd.crosstab(df.DayOfWeek, df.Category)
data = ct.values

# probability of each crime category by DayOfWeek
p = data * 1.0 /data.sum()
# probability of each day (sum of all crimes)
p_ = p.sum(axis=1)
# entropy and conditional entropy are used to calculate the information gain
entropy(p_) 
conditional_entropy(p)
information_gain(p) # 0.005

#information gain category,hour
ct2 = pd.crosstab(df.Hour, df.Category)
data2 = ct2.values
p2 = data2 * 1.0 /data2.sum()
information_gain(p2) # 0.05


#information gain category, district
ct3 = pd.crosstab(df.PdDistrict, df.Category)
data3 = ct3.values
p3 = data3 * 1.0 /data3.sum()
information_gain(p3) # 0.095

#information gain category, month
ct4 = pd.crosstab(df.Month, df.Category)
data4 = ct4.values
p4 = data4 * 1.0 /data4.sum()
information_gain(p4) # 0.002



