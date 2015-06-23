import pandas as pd
import numpy as np
from matplotlib import pylab, mlab, pyplot
plt = pyplot

df = pd.read_csv("../crime_train.csv")
df = df.reindex(np.random.permutation(df.index))
# TODO: now remove the test data!

def getYear(s):
  return s.split("-")[0]
def getMonth(s):
  return s.split("-")[1]

df['Year']= df['Dates'].apply(lambda x: getYear(x))
df['Month']= df['Dates'].apply(lambda x: getMonth(x))
df['Year'] = df['Year'].apply(int)
df['Month'] = df['Month'].apply(int)
# remove the data for the year 2015, as the year is not yet over
df = df[df.Year != 2015]


years = df.Year.unique()
years
ct = pd.crosstab(df.Category, df.Year)
a = ct.ix[23:24].T
a
a.describe()
a.plot()

crime_by_year = df.groupby('Year').Category.count()
crime_by_month = df.groupby('Month').Category.count()




# df['Month'] = df['Month'].apply(int)
seasons = ['Winter', 'Spring', 'Summer', 'Fall']
month_bins = pd.cut(df.Month, [df.Month.min(), 4, 7, 10, df.Month.max()], labels = seasons)

data = df.groupby([month_bins,'Month']).Category.count()
data2 = df.groupby([month_bins,'DayOfWeek']).Category.count()

groups = df.groupby(month_bins)
seasons_view = groups.Category.count()
seasons_view
seasons_view.plot(kind='bar')
data.unstack(level=1).plot(kind='bar', subplots=False, ax=ax)

fig = pyplot.figure(figsize=(8,4))
ax = fig.add_subplot(111)
ax.set_xlabel('Age')
ax.set_ylabel('Count of foo')
ax.set_title("Count of foo by bar and bla")
data2.unstack(level=1).plot(kind='bar', subplots=False, ax=ax)


var = df.groupby(['DayOfWeek','Year']).Month.count()
ax = var.unstack().plot(kind='bar',stacked=True,  color=['red','blue','green', 'yellow', 'white'
, 'black', 'orange', 'pink', 'purple', 'brown', 'grey', 'gold'], grid=False)
ax.set_xlabel('Age')
ax.set_ylabel('Number of Passengers')

var = df.groupby(['Year','DayOfWeek']).Month.count()
ax = var.unstack().plot(kind='bar',stacked=True,  color=['red','blue','green', 'yellow', 'orange', 'pink', 'purple', 'white'], grid=False)
ax.set_xlabel('Year')
ax.set_ylabel('Number of Crimes')
data = df.groupby([month_bins,'Year']).Category.count()
data
data2 = df.groupby([month_bins,'DayOfWeek']).Category.count()
data2



# things to work on:
'''
df_nn = df[pd.notnull(df['Year'])]
import seaborn as sns
sns.violinplot(df_nn['Year'], df_nn['Category'].head(), range = (df['Year'].min(),df['Year'].max())) 
sns.despine()
'''

'''
from statsmodels.graphics.mosaicplot import mosaic

mosaic(df.head(15), ['DayOfWeek', 'Month', 'Category'])

df['Adult'] = df["Age"].apply(lambda age: "adult" if age >14. else "child")
#or df['Adult'] = df["Age"]>14.
mosaic(df, ['Survived', 'Sex', 'Pclass', 'Adult'])

#) Probability of surviving of a woman in the 3. class?
df[(df['Category']=='PROSTITUTION') & (df['Year']==2013)].Year.mean()
'''


'''
df['Year'].apply(int)
df['Month'].apply(int)
temp2 = df.groupby('Year').Month.sum()/df.groupby('Year').DayOfWeek.count()
fig = pyplot.figure(figsize=(8,4))
ax1 = fig.add_subplot(121)
ax1.set_xlabel('Pclass')
ax1.set_ylabel('Count of Passengers')
ax1.set_title("Passengers by Pclass")
temp1.plot(kind='bar')

ax2 = fig.add_subplot(122)
temp2.plot(kind = 'bar')
ax2.set_xlabel('Pclass')
ax2.set_ylabel('Probability of Survival')
ax2.set_title("Probability of survival by class")
'''


'''
pd.to_datetime(df['Dates'])

dt = df['Dates'].to_frame()
a = pd.to_datetime(df['Dates'])
dt = a.to_frame()
'''



