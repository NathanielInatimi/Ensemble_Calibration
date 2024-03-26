# -*- coding: utf-8 -*-
"""
function to read in the ACE multi-instrument HDF files

from: ftp://cdaweb.gsfc.nasa.gov/pub/data/ace/multi

@author: vy902033
"""

from pyhdf.SD import SD, SDC


# Open file.
FILE_NAME = 'C:\\Users\\vy902033\\Dropbox\\Data\\ACEcdf\\multi\\multi_data_1hr_year2000.hdf'
file = SD(FILE_NAME, SDC.READ)

datasets_dic = file.datasets()

# List available SDS datasets.
print(str(file.datasets()))
