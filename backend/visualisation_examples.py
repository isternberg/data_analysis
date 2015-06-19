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




