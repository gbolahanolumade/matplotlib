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