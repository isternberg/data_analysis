import pandas as pd
import numpy as np
from matplotlib import pylab, mlab, pyplot
plt = pyplot

df_train = pd.read_csv("../df_train.csv")


# count of each crime category by day of week
ct = pd.crosstab(df_train.Category, df_train.DayOfWeek)
display = ct.ix[:,['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']] 
display
# count of prostitution cases by day (for the entire period 2003 - 2014)
prostitution_row = display.ix[23:24] 
prostitution_row
# statistical data to prostitution (by day)
prostitution_row.T.describe()

# TODO add the threshhold of avg
# Crime in San Fransico 2003 - 2014 by district
crimes_count_by_district = df_train.groupby('PdDistrict').Category.count().to_frame()
crimes_count_by_district.columns=["Sum_of_crimes"]
crimes_count_by_district.sort(columns="Sum_of_crimes",ascending=1)["Sum_of_crimes"].plot(kind="bar", 
title='Crime in San Fransico 2003 - 2014 by district')
plt.xlabel("district") 
plt.ylabel("count") 
plt.tight_layout()
plt.show()

# Prostitution in San Fransico 2003 - 2014
ct = pd.crosstab(df_train.Category, df_train.Year)
prostitution = ct.ix[23:24].T
prostitution
prostitution.describe()
ax = prostitution.plot(lw=2,colormap='jet',marker='.',markersize=10,title='Prostitution in San Fransico 2003 - 2014')
ax.set_ylabel("count")
ax.set_xlabel("year")
plt.show()
#plt.savefig("_plotname")

# Crime in San Fransico 2003 - 2014
crime_by_year = df_train.groupby('Year').Category.count()
ax = crime_by_year.plot(lw=2,colormap='jet',marker='.',markersize=10,title='Crime in San Fransico 2003 - 2014')
ax.set_ylabel("count")
ax.set_xlabel("year")
pylab.ylim(40000, 70000)
plt.show()

#TODO replace month number with month name
crime_by_month = df_train.groupby('Month').Category.count()
ax = crime_by_month.plot(kind='bar',title='Crime in San Fransico by month (2003-2014)')
ax.set_ylabel("count")
ax.set_xlabel("month")
plt.show()

# Corr and Cov TODO check for correctness
ct = pd.crosstab(df_train.Year, df_train.Category)
ct.DRUNKENNESS.corr(ct.VANDALISM)
ct.DRUNKENNESS.cov(ct.VANDALISM)

# replaces with bar plot, 3 strongest corrs
ct.corr().head(3).T
ct.cov().head(3).T
ct.corrwith(ct.VANDALISM)
ax = ct.corr().head(5).T[5:10].plot(kind='bar',colormap='jet',
title='Correlation between 5 crimes (2003-2014)')
ax.set_ylabel("correlation")
ax.set_xlabel("category")
pylab.ylim([-1,1])
plt.tight_layout(pad= .1, w_pad=3.8, h_pad=1.0)# fix
plt.show()


# could be nice to understand this plot someday
%pylab qt4
plt.scatter(ct.DRUNKENNESS, ct.VANDALISM) 
plt.xlabel('change of Drunkness??')
plt.ylabel('change of vandalism??')


# df['Month'] = df['Month'].apply(int)
seasons = ['Winter', 'Spring', 'Summer', 'Fall']
month_bins = pd.cut(df_train.Month, [0, 3, 6, 9, df_train.Month.max()], labels = seasons)
data = df_train.groupby([month_bins,'Month']).Category.count()
data2 = df_train.groupby([month_bins,'DayOfWeek']).Category.count()
# TODO: rename columns

groups = df_train.groupby(month_bins)
seasons_view = groups.Category.count()
seasons_view
ax = seasons_view.plot(kind='bar',title='Crime in San Fransico by season (2003-2014)')
ax.set_ylabel("count")
ax.set_xlabel("season")
plt.tight_layout()
plt.show()

fig = pyplot.figure(figsize=(8,4))
ax = fig.add_subplot(111)
ax.set_ylabel("count")
ax.set_xlabel("season")  # how come xlabel in plot = month?
data2.unstack(level=1).plot(kind='bar', subplots=False, ax=ax, title="Crime in San Fransico by season and day (2003-2014)")
plt.tight_layout()
plt.show()

#inspiration for plot: https://www.kaggle.com/ldocao/sf-crime/exploratory-horizontal-bar-plots
categories = df.groupby("Category")
count = categories.count()
plt.figure()
plt.xlabel("count")
count.sort(columns="X",ascending=1)["X"].plot(kind="barh") 
plt.tight_layout() 
plt.show()
#plt.savefig("categories_count")

#maybe we could do smth like this?
import matplotlib.dates as mdates
from mpl_toolkits.mplot3d import Axes3D
matplotlib.rcParams['font.size'] = 10;
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for z in [df_train.Year.unique()]:
  xs = xrange(1,13)
  ys = 1000 * np.random.rand(12)
  color = plt.cm.Set2(random.choice(xrange(plt.cm.Set2.N)))
  ax.bar(xs, ys,zs=z, zdir='y', color=color, alpha=0.8)

ax.set_xlabel('Month')
ax.set_ylabel('Year')
ax.set_zlabel('Sales')
plt.show()

# Pie Chartx = 
from pylab import *
# start_x, start_y, width, height
ax = axes([0.1,0.1, 0.7,0.7])
labels = 'burglary', 'theft', 'murder', 'assult'
x = [15, 30, 45, 10]
# emphasis of each data
explode = (0.1, 0.1, 0.1, 0.3)
# autopct = position of values
pie(x, explode, labels, autopct='%4.1f%%', startangle=67)
title('Crimes on Mondays')
show()




