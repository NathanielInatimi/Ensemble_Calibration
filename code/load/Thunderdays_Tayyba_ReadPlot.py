# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 08:47:52 2017

@author: vy902033
"""

import os as os
import numpy as np
import matplotlib.pyplot as plt
#Load and plot thunder data

datapath = os.environ['DBOX'] + 'Papers_WIP\\Thunder_LongTerm\\TayybaAhkter\\Python\\'
startyr=1852
stopyr=1861
yspacing=15


#set up the plot
fig, ax = plt.subplots(1, 1, figsize=(12, 15))
# Remove the plot frame lines. They are unnecessary here.
ax.spines['top'].set_visible(False)
#ax.spines['bottom'].set_visible(False)
ax.spines['right'].set_visible(False)
#ax.spines['left'].set_visible(False)

# Ensure that the axis ticks only show up on the bottom and left of the plot.
# Ticks on the right and top of the plot are generally unnecessary.
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()

#fig.subplots_adjust(left=.06, right=.75, bottom=.02, top=.94)
# Limit the range of the plot to only where the data is.
# Avoid unnecessary whitespace.
ax.set_xlim(0.5, 12.5)
#ax.set_ylim(-0.25, 90)

# Make sure your axis ticks are large enough to be easily read.
# You don't want your viewers squinting to read your plot.
plt.xticks(range(1, 13, 1), fontsize=18)
plt.yticks(range(0, 0, 10), fontsize=18)
ax.xaxis.set_major_formatter(plt.FuncFormatter('{:.0f}'.format))
ax.yaxis.set_major_formatter(plt.FuncFormatter('{:.0f}%'.format))

# These are the colors that will be used in the plot
color_sequence = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c',
                  '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5',
                  '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f',
                  '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5']

count=0

#plot a little "size" marker for thunderdays
nyrs=stopyr-startyr
arr=plt.arrow(1.5, yspacing*nyrs+yspacing/2, 0, 20,lw=2.5,head_width=0.1, head_length=1, fc='k', ec='k')
ax.add_patch(arr)
arr=plt.arrow(1.5, yspacing*nyrs+yspacing/2 +20, 0, -20,lw=2.5,head_width=0.1, head_length=1, fc='k', ec='k')
ax.add_patch(arr)
line = plt.plot([1.35,1.65],[yspacing*nyrs+yspacing/2-1,yspacing*nyrs+yspacing/2-1], lw=2.5,color='k')
line = plt.plot([1.35,1.65],[yspacing*nyrs+yspacing/2+21,yspacing*nyrs+yspacing/2+21], lw=2.5,color='k')
plt.text(1.64,yspacing*nyrs+yspacing/2 +10 , '20',fontsize=18, color='k')
ax.set_ylabel('Monthly thunderdays',fontsize=18)

#change the x-axis ticks
my_xticks = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct', 'Nov', 'Dec']
plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12], my_xticks)

for yr in range(startyr,stopyr+1):
    #print(str(yr))
    
    #LOAD THE DATA
    data=np.genfromtxt(datapath+'data'+str(yr)+'.txt')
    
    line = plt.plot(data[:,0],
                    data[:,1]+count*yspacing,
                    lw=2.5,
                    color=color_sequence[count])
    
    plt.text(12.3, data[11,1]+count*yspacing, str(yr),fontsize=18, color=color_sequence[count])
    
    count = count + 1
    
#data= np.genfromtxt(datapath)