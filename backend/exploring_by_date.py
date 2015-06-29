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
df_train.to_csv("../df_train.csv",  encoding='utf-8')
df_test.to_csv("../df_test.csv",  encoding='utf-8')


ct = pd.crosstab(df_train.Category, df_train.Year)
prostitution = ct.ix[23:24].T
prostitution
prostitution.describe()
ax = prostitution.plot(lw=2,colormap='jet',marker='.',markersize=10,title='Prostitution in San Fransico 2003 - 2014')
ax.set_ylabel("count")
ax.set_xlabel("year")

crime_by_year = df_train.groupby('Year').Category.count()
ax = crime_by_year.plot(lw=2,colormap='jet',marker='.',markersize=10,title='Crime in San Fransico 2003 - 2014')
ax.set_ylabel("count")
ax.set_xlabel("year")

crime_by_month = df_train.groupby('Month').Category.count()
ax = crime_by_month.plot(kind='bar',title='Crime in San Fransico by month (2003-2014)')
ax.set_ylabel("count")
ax.set_xlabel("month")



# df['Month'] = df['Month'].apply(int)
seasons = ['Winter', 'Spring', 'Summer', 'Fall']
month_bins = pd.cut(df_train.Month, [df_train.Month.min(), 4, 7, 10, df_train.Month.max()], labels = seasons)

data = df_train.groupby([month_bins,'Month']).Category.count()
data2 = df_train.groupby([month_bins,'DayOfWeek']).Category.count()

groups = df_train.groupby(month_bins)
seasons_view = groups.Category.count()
seasons_view
ax = seasons_view.plot(kind='bar',title='Crime in San Fransico by season (2003-2014)')
ax.set_ylabel("count")
ax.set_xlabel("season")
plt.tight_layout()


fig = pyplot.figure(figsize=(8,4))
ax = fig.add_subplot(111)
ax.set_ylabel("count")
ax.set_xlabel("season")  # how come xlabel in plot = month?
data2.unstack(level=1).plot(kind='bar', subplots=False, ax=ax, title="Crime in San Fransico by season and day (2003-2014)")
plt.tight_layout()

#inspiration for plot: https://www.kaggle.com/ldocao/sf-crime/exploratory-horizontal-bar-plots
categories = df.groupby("Category")
count = categories.count()
plt.figure()
plt.xlabel("count")
count.sort(columns="X",ascending=1)["X"].plot(kind="barh") 
plt.tight_layout() 
#plt.savefig("categories_count")







