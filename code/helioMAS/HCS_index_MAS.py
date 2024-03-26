# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 13:29:38 2022

@author: mathewjowens
"""

#compute the current sheet tilt index from MAS data

import httplib2
import urllib
import os
from pyhdf.SD import SD, SDC  
import numpy as np
import astropy.units as u
import ssl
import matplotlib.pyplot as plt

import ReadMAS as mas

datadir = os.environ['DBOX'] + 'Data\\'
masdir = datadir + 'MAS_HI\\'



CRstart = 1625
CRstop = 2260
observatories = ['hmi', 'mdi', 'solis', 'gong', 'mwo', 'wso', 'kpo']

r_vals = [1.0, 1.1, 1.25, 1.5, 2, 3, 5, 10, 30]

nr = len(r_vals)
nCR = CRstop - CRstart + 1
ihcs_r_CR = np.ones((nr, nCR)) * np.nan
mas_res = np.ones((3, nCR)) * np.nan

iCR = 0
for CR in range(CRstart,CRstop):
    
    
    #see if there's any data, use observatories in order of preference
    found_mas = False
    for obs in observatories:
        filepath = masdir + obs + '\\CR' + str(CR) + '\\br002.hdf'
        if (os.path.exists(filepath) == True):
            MAS_br, MAS_br_Xa, MAS_br_Xm, MAS_br_Xr = mas.read_HelioMAS(filepath)
            print('Processing: ' + filepath)
            found_mas = True
            break
        
    if found_mas:
        mas_res[0,iCR] = len(MAS_br_Xr)
        mas_res[1,iCR] = len(MAS_br_Xm)
        mas_res[2,iCR] = len(MAS_br_Xa)
    
        for ir in range(0,nr):
            
            r = r_vals[ir]
            r_index = np.argmin(abs(MAS_br_Xr.value - r))
            
            ihcs = np.ones(len(MAS_br_Xm)) * np.nan
            for ilat in range(0,len(MAS_br_Xm)):
                #convert the long profile into +/- 1
                bpol = np.sign(MAS_br[:, ilat, r_index])
                #count how many changes in polarity are present.
                dbpol = np.abs(bpol[:-1] - bpol[1:])/2
                nswitch = dbpol.sum()
                #check the end points too
                if not dbpol[-1] == dbpol[0]:
                    nswitch = nswitch + 1
                
                #save the answer for this lat
                ihcs[ilat] = nswitch / len(bpol)
                
            #compute the lat-weighted iHCS
            ihcs_r_CR[ir, iCR] = np.mean(np.sin(MAS_br_Xm) * ihcs)
    else:
        ihcs_r_CR[:, iCR] = np.nan
            
    iCR = iCR + 1
        

plt.figure()

plt.subplot(2,1,1)
plt.plot(ihcs_r_CR[1,:])
plt.plot(ihcs_r_CR[3,:])
plt.plot(ihcs_r_CR[5,:])

plt.subplot(2,1,2)
plt.plot(mas_res[0,:], label = 'N_R')
plt.plot(mas_res[1,:], label = 'N_lon')
plt.plot(mas_res[2,:], label = 'N_lat')
