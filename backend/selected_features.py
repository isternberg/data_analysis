import pandas as pd
import numpy as np
from matplotlib import pylab, mlab, pyplot
plt = pyplot

df = pd.read_csv("../df_train.csv")

#use only PdDistrict and Hour for prediction, since
#they have the highest information gain.
reduced_training_data = df.iloc[:,[5, 12, 2]]
