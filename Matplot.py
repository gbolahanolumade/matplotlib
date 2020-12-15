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