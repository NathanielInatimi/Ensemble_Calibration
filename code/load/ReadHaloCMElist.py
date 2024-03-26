# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 16:12:12 2017

A script to read and process teh CDAW Halo CME list.

Some pre-processing is required:
    Download the following webpage as a html file: 
        https://cdaw.gsfc.nasa.gov/CME_list/halo/halo.html
    Open in Excel delete last two columns (J and K) and the bottom rows
    Save as a CSV.

@author: vy902033
"""
import pandas as pd
import os as os
import numpy as np
import re  #for deadling with non-numeric characters in a string of unknown length
from datetime import datetime
import matplotlib.pyplot as plt

def HaloCMElist():
    filepath= os.environ['DBOX'] + 'Data\\CME_list\\SOHO_LASCO HALO CME Catalog.csv'
    
    halocmes=pd.read_csv(filepath,header=None)
    #delete the first few rows
    halocmes.drop(halocmes.index[:11], inplace=True)
    #reindex
    halocmes.index = range(len(halocmes))
    
    for rownum in range(0,len(halocmes)):
        #convert the  date stamp - first put the date and time together
        datestr=halocmes[0][rownum] + ' ' + halocmes[1][rownum]
        #compute the datetime object. The 2017 dates lack seconds (because why would you keep the same format?)
        if (len(halocmes[1][rownum])>5):
            datetime_obj = datetime.strptime(datestr, '%d-%m-%y %H:%M:%S')
        else:
            datetime_obj = datetime.strptime(datestr, '%d-%m-%y %H:%M')
        #print(datetime_obj)
        halocmes.set_value(rownum,0,datetime_obj)
        
        #tidy up the remaining properties, datagap normally starts with --
        aspeed=str(halocmes[2][rownum])  #apparent spped (plane of sky)
        if (aspeed[:2]== '--'):
            halocmes.set_value(rownum,2,np.nan)
        else:
            halocmes.set_value(rownum,2,float(aspeed))
            
        sspeed=str(halocmes[3][rownum]) #space speed - from a cone model?
        if (sspeed[:2]== '--'):
            halocmes.set_value(rownum,3,np.nan)
        else:
            halocmes.set_value(rownum,3,float(sspeed))
        if halocmes[3][rownum]>8000: #there are some crazy values in the space speeds
            halocmes.set_value(rownum,3,np.nan)
            
        #source location - 1 for frontside, 0 for backside
        slocation=str(halocmes[6][rownum])
        if (slocation[:8]=='Backside'):
            halocmes.set_value(rownum,6,0)
        elif (slocation[:1]=='N' or slocation[:1]=='S' or
              slocation[:1]=='E' or slocation[:1]=='W'):
            halocmes.set_value(rownum,6,1) 
        else:
            halocmes.set_value(rownum,6,np.nan)
        
        #position angle (biut useless for halos)
        halocmes.set_value(rownum,5,float(halocmes[5][rownum]))
        
        #flare magnitude
        fmag=str(halocmes[7][rownum])
        if (fmag[:2]=='--'):
            halocmes.set_value(rownum,7,np.nan)
        elif(fmag[:1]=='A'):
            halocmes.set_value(rownum,7,float(fmag[1:]))
        elif(fmag[:1]=='B'):
            halocmes.set_value(rownum,7,float(fmag[1:])+10)
        elif(fmag[:1]=='C'):
            halocmes.set_value(rownum,7,float(fmag[1:])+20)
        elif(fmag[:1]=='M'):
            halocmes.set_value(rownum,7,float(fmag[1:])+30)
        elif(fmag[:1]=='X'):
            halocmes.set_value(rownum,7,float(fmag[1:])+40)
                   
    #remove the time, acc and flare time columns
    del halocmes[1]    
    del halocmes[4]
    del halocmes[8]          
        
        
    
    #chage teh column headings
    halocmes=halocmes.rename(columns = {0:'CME_time',
                                  2:'v_pos',
                                  3:'v_space',
                                  5:'MPA',
                                  6: 'front_flag',
                                  7:'flare_mag'})
    return halocmes
             
