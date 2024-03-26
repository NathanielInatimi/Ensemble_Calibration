# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 16:22:53 2023

@author: mathewjowens
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 16:12:10 2022

A script to plot STEREO and OMNI solar wind observations as a function of helio
lat and long


@author: mathewjowens
"""



import numpy as np
import pandas as pd
import astropy.units as u
from astropy.time import Time, TimeDelta
import os
import glob
import h5py
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from matplotlib.lines import Line2D
import datetime as datetime
from scipy import interpolate

import conversions.helio_time as htime
import conversions.helio_coords as hcoord


data_dir = os.path.join(os.environ['DBOX'], 'Data_hdf5')
ephem_dir = os.path.join(os.environ['DBOX'],'python_repos','HUXt','data','ephemeris')

latscan = 1
res_hr = 8




#set the dates of the fast latitude scans
if latscan==1:
    smjd=49607
    fmjd=49930
elif latscan==2:
    smjd=51872
    fmjd=52194
elif latscan==3:
    smjd=54139
    fmjd=54479





Ulysses_1hour = pd.read_hdf(os.path.join(data_dir,'ulysses_1hour.h5'))

ephemeris = h5py.File(os.path.join(ephem_dir, 'ephemeris.hdf5'), 'r')
E_Carr = ephemeris['EARTH']['CARR']['longitude'][...]*np.pi/180
E_HAE = hcoord.zerototwopi(ephemeris['EARTH']['HAE']['longitude'][...]*np.pi/180)
E_time = Time(ephemeris['EARTH']['HEEQ']['time'], format='jd').mjd



res_str = str(res_hr) + 'H'

#average data to 1-day resolution
Ulysses_1hour.set_index('datetime', inplace=True)
Ulysses = Ulysses_1hour.resample(res_str).mean()
#correct for PANDAS insane time stamp butchering
Ulysses.index = Ulysses.index + datetime.timedelta(hours=res_hr/2)

Ulysses['SC_hlong'] = Ulysses['SC_hlong'] *np.pi/180
Ulysses['SC_hlat'] = Ulysses['SC_hlat'] *np.pi/180

#recompute Carr-lon to avoid 0/2pi mixing
#f = interpolate.interp1d( Ulysses_1hour['mjd'], Ulysses_1hour['Carr_lon'], fill_value = np.nan, kind = 'nearest')
#Ulysses['Carr_lon'] = f(Ulysses['mjd'])

#interpolate the Earth Carr and HAE longs onto the Ulysses time step, avoiding 0/2pi mixing
f = interpolate.interp1d( E_time, E_Carr, fill_value = np.nan, kind = 'nearest')
Ulysses['Earth_Carr_lon'] = f(Ulysses['mjd'])
f = interpolate.interp1d( E_time, E_HAE, fill_value = np.nan, kind = 'nearest')
Ulysses['Earth_HAE_lon'] = f(Ulysses['mjd'])
#Compute Ulysses Carr long from Earth's
dphi = Ulysses['SC_hlong'] - Ulysses['Earth_HAE_lon']
Ulysses['SC_Clon'] =  hcoord.zerototwopi(Ulysses['Earth_Carr_lon'] + dphi)

#get the datetime variable from the index
Ulysses.reset_index()

#cut out the datachunk of interest
mask = (Ulysses['mjd'] >= smjd) & (Ulysses['mjd'] < fmjd)
datachunk = Ulysses.loc[mask] 
datachunk.reset_index()


# <codecell>


plt.rcParams.update({
    "text.usetex": False,
    'font.size': 14,
    "font.sans-serif": ["Helvetica"]})


vmax = 750
vmin = 250



fig = plt.figure(figsize = (8,8))

ax = plt.subplot(111)


    
    
    
alpha = 1#min_alpha + (max_alpha - min_alpha)*(counter/ndays)
point_size = 100# min_point_size + (max_point_size - min_point_size)*(counter/ndays)
#print(t)

#find the closest time to that required
for t in range(0, len(datachunk)):
                    
    im1 = ax.scatter(datachunk['SC_Clon'][t]*180/np.pi, datachunk['SC_hlat'][t],  
                s = point_size, c = datachunk['Vr'][t], alpha = alpha, edgecolors='none',
                marker = 'o')
    im1.set_clim(vmin,vmax)
            


# cbar = fig.colorbar(im2, ax=ax)
# cbar.set_label(r'$V$ [km/s]')
# ax3.set_ylim(-10,10)
# ax3.set_xlim(0,360)
# ax3.set_xticks([0,90,180,270,360])
# ax3.set_ylabel(r'Latitude [$^\circ$]')
# ax3.set_xlabel(r'Solar longitude [$^\circ$]')

# if figcounter <10:
#     fignum = '0000' +str(figcounter)
# elif figcounter <100:
#     fignum = '000' +str(figcounter)
# elif figcounter <1000:
#     fignum = '00' +str(figcounter)
# elif figcounter <10000:
#     fignum = '0' +str(figcounter)

# plt.savefig('fig_lo_'+ fignum + '.png', format = 'png')
# plt.close(fig)
# figcounter = figcounter +1


