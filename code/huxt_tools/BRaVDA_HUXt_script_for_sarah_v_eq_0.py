# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 14:41:43 2023

@author: mathe
"""

from astropy.time import Time, TimeDelta
import numpy as np
import datetime
import os as os
import matplotlib.pyplot as plt


#from BRaVDA
import startBravda






#bravda ensemble directory
bravda_dir = os.environ['DBOX'] + 'python_repos\\BRaVDA\\'


#DA window dates
starttime = datetime.datetime(2021,12,3)
endtime = starttime + datetime.timedelta(days=27.27)

obs = ['STERA', 'OMNI']#  ['OMNI'] #


startBravda.bravdafunction(endtime, obsToAssim = obs , usecustomens = True,
                           plottimeseries = True,
                           precondState = False, useLogTrans = False)

#read in the posterior solution 
smjd = int(Time(starttime).mjd)
posterior = np.loadtxt(bravda_dir + 'test\\posterior\\posterior_MJDstart' + str(smjd) +'.txt')


plt.figure()
plt.plot(posterior[0,:])
