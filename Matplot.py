# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 18:45:18 2020

@author: GbolahanOlumade
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.plot([1,2,3,4,5])
plt.show

plt.plot([1,2,3,4,5], [4,5,6,7,8])
plt.show

plt.axis([0,5,0,20])
plt.title('my first work')
plt.plot([1,2,3,4],[1,4,9,16], 'ro')

import math

t = np.arange(0,2.5,0.1)
t
y1 = list(map(math.sin,math.pi*t))
y2 = list(map(math.sin,math.pi*t+math.pi/2))
y3 = list(map(math.sin,math.pi*t-math.pi/2))
plt.plot(t,y1,'b*',t,y2,'g^',t,y3,'ys')

plt.plot(t,y1,'b--',t,y2,'g',t,y3,'r-.')

math.sin(t)
plt.plot([1,2,4,2,1,0,1,2,1,4],linewidth=2.0)

t = np.arange(0,5,0.1)
y1 = np.sin(2*np.pi*t)
y2 = np.sin(2*np.pi*t)
plt.subplot(211)
plt.plot(t,y1,'b-.')
plt.subplot(212)
plt.plot(t,y2,'r--')


plt.axis([0,5,0,20])
plt.title('My first plot')
plt.xlabel('Counting')
plt.ylabel('Square Values')
plt.plot([1,2,3,4],[1,4,9,16], 'ro')

plt.axis([0,5,0,20])
plt.title('My first plot', fontsize=20, fontname='Times New Roman')
plt.xlabel('Counting', color='gray')
plt.ylabel('Square', color='gray')
plt.plot([1,2,3,4],[1,4,9,16], 'ro')


plt.axis([0,5,0,20])
plt.title('My first plot', fontsize=20, fontname='Times New Roman')
plt.xlabel('Counting', color='gray')
plt.ylabel('Square', color='gray')
plt.text(1,1.5, 'First')
plt.text(2,4.5, 'Second')
plt.text(3,9.5, 'Third')
plt.text(4,16.5, 'Third')
plt.plot([1,2,3,4],[1,4,9,16], 'ro')

import datetime
events = [datetime.date(2015,1,23),datetime.date(2015,1,28),datetime.
date(2015,2,3),datetime.date(2015,2,21),datetime.date(2015,3,15),datetime.
date(2015,3,24),datetime.date(2015,4,8),datetime.date(2015,4,24)]
readings = [12,22,25,20,18,15,17,14]
plt.plot(events,readings)

import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
months = mdates.MonthLocator()
days = mdates.DayLocator()
timeFmt = mdates.DateFormatter('%Y-%m')
events = [datetime.date(2015,1,23),datetime.date(2015,1,28),datetime.
date(2015,2,3),datetime.date(2015,2,21),datetime.date(2015,3,15),datetime.
date(2015,3,24),datetime.date(2015,4,8),datetime.date(2015,4,24)]
readings = [12,22,25,20,18,15,17,14]
fig, ax = plt.subplots()
plt.plot(events,readings)
ax.xaxis.set_major_locator(months)
ax.xaxis.set_major_formatter(timeFmt)
ax.xaxis.set_minor_locator(days)

pop = np.random.randint(0,100,100)
n,bins,patches = plt.hist(pop,bins=20)


index = np.arange(5)
values = [5,7,3,4,6]

plt.bar(index,values)
plt.xticks(index+0.4,['A','B','C','D','E'])

index = np.arange(5)
values1 = [5,7,3,4,6]
std1 = [0.8,1,0.4,0.9,1.3]
plt.title('A Bar Chart')
plt.bar(index,values1,yerr=std1,error_kw={'ecolor':'0.1',
'capsize':6},alpha=0.7,label='First')
plt.xticks(index+0.4,['A','B','C','D','E'])
plt.legend(loc=2)


import matplotlib.pyplot as plt
import numpy as np
index = np.arange(5)
values1 = [5,7,3,4,6]
values2 = [6,6,4,5,7]
values3 = [5,6,5,4,6]
bw = 0.3
plt.axis([0,8,0,5])
plt.title('A Multiseries Horizontal Bar Chart',fontsize=20)
plt.barh(index,values1,bw,color='b')
plt.barh(index+bw,values2,bw,color='g')
plt.barh(index+2*bw,values3,bw,color='r')
plt.yticks(index+0.4,['A','B','C','D','E'])

import matplotlib.pyplot as plt
import pandas as pd
data = {'series1':[1,3,4,3,5],
'series2':[2,4,5,2,4],
'series3':[3,2,3,1,3]}
df = pd.DataFrame(data)
df.plot(kind='bar', stacked=True)

import matplotlib.pyplot as plt
labels = ['Nokia','Samsung','Apple','Lumia']
values = [10,30,45,15]
colors = ['yellow','green','red','blue']
plt.pie(values,labels=labels,colors=colors)
plt.axis('equal')


import matplotlib.pyplot as plt
labels = ['Nokia','Samsung','Apple','Lumia']
values = [10,30,45,15]
colors = ['yellow','green','red','blue']
explode = [0.3,0.2,0,0]
plt.title('A Pie Chart')
plt.pie(values,labels=labels,colors=colors,explode=explode,startangle=180)
plt.axis('equal')

import matplotlib.pyplot as plt
import numpy as np
dx = 0.01; dy = 0.01
x = np.arange(-2.0,2.0,dx)
y = np.arange(-2.0,2.0,dy)
X,Y = np.meshgrid(x,y)
def f(x,y):
 return (1 - y**5 + x**5)*np.exp(-x**2-y**2)
C = plt.contour(X,Y,f(X,Y),8,colors='black')
plt.contourf(X,Y,f(X,Y),8)
plt.clabel(C, inline=1, fontsize=10)
    
x = np.arange(5)
y = (20,35,30,37,40)

plt.bar(x,y)
plt.show()
plt.scatter(x,y)


x = np.linspace(0,20,1000)
y = np.sin(x)

plt.plot(x,y)
plt.title('Sample Plot Title')
plt.grid(True)



plt.plot([1,2,3,4,]);


fig = plt.figure()
ax = fig.add_subplot()
plt.show() 

x= [1,2,3,4]
y= [1,2,3,4]

fig = plt.figure()
ax = fig.add_axes([1,1,1,1])
ax.plot(x,y)
plt.show(),


fig, ax = plt.subplots()
ax.plot(x,y);
type(fig), type(ax)


fig, ax= plt.subplots(figsize=(10,10))
ax.plot(x,y)
ax.set(title='Simple Plot', xlabel='X-axis', ylabel='y-axis')
