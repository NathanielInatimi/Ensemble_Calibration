# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 10:52:55 2022

@author: vy902033
"""
import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

filepath = 'Dst_1972-10-19_2016-12-31_D.dat'
data = pd.read_csv(filepath, delim_whitespace=True, header = 24)



dfdt=np.empty_like(data['DATE'],dtype=datetime)
for i in range(0,len(data)):
    date_time_str = data['DATE'][i] + ' ' + data['TIME'][i]
   
    format = "%Y-%m-%d %H:%M:%S.%f"
    dfdt[i] = datetime.strptime(date_time_str, format)

data['datetime'] = dfdt

plt.plot(data['datetime'],data['Dst'])