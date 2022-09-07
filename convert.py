# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 21:28:22 2022

@author: Administrator
"""

import numpy as np

import pandas as pd

import xarray as xr

import glob

from wrf import getvar,pvo,interplevel,destagger,latlon_coords

from netCDF4 import Dataset

from metpy.units import units
import metpy.constants as constants
import metpy.calc as mpcalc
'''
This script is used to convert vertical wind speed from m/s to pa/s
'''

###################################################################################
######   read wrf data path
###################################################################################
path  = r'/Users/xpji/out0-8/'
filelist = glob.glob(path+'wrf*')
filelist.sort()

###################################################################################
######  get longitude,latitude,dx,dy
###################################################################################
def  cal_dxdy(file):
    ncfile = Dataset(file)
    P = getvar(ncfile, "pressure")
    lats, lons = latlon_coords(P)
    lon = lons[0]
    lon[lon<=0]=lon[lon<=0]+360
    lat = lats[:,0]
    dx, dy = mpcalc.lat_lon_grid_deltas(lon.data, lat.data)
    return lon,lat,dx,dy

lon,lat,dx,dy = cal_dxdy(filelist[-1]) 
nx = lon.shape[0]
ny = lat.shape[0]
###################################################################################
######  define function to convert m/s to pa/s
###################################################################################
def cal_omega(file):
    
    ncfile = Dataset(file)
    
    pre = getvar(ncfile,'pressure')
    
    w = getvar(ncfile,'omega')

    
    level =np.array( [100,125,
                      150,175,200,
                      225,250,300,
                      350,400,450,
                      500,550,600,
                      650,700,750,
                      775,800,825,
                      850,875,900,
                      925,950,975,
                      1000]
            )
    level=level[::-1]
    
    omega = interplevel(w,pre,level)
    
    return omega
###################################################################################
######  converting
###################################################################################
omega = np.full([len(filelist),27,ny,nx],np.nan,dtype='float32')

for i in range(len(filelist)):
    print(i)
    omega[i,:,:,:]       =  cal_omega(filelist[i])
###################################################################################
######  output data as Netcdf format
###################################################################################
time=pd.date_range(start='2004-06-21-00',end='2004-06-30-18',freq='6H').strftime('%Y_%m_%d_%H:%M:%S')
lev=np.array([100,125,
              150,175,200,
              225,250,300,
              350,400,450,
              500,550,600,
              650,700,750,
              775,800,825,
              850,875,900,
              925,950,975,
              1000])
lev =lev[::-1]
da_nc=xr.Dataset(
data_vars=dict(
    
     omega=(['time','level','lat','lon'],np.array(omega)),

     ),      
                 coords = {
                     'time':time,
                     'level':lev.data,
                     'lat':lat.data,
                     'lon':lon.data,
                     
                     },
                 attrs=dict(description="vertical motion",
                            units= 'pa/s'),
                 )
da_nc.to_netcdf(r'omega.nc') 
 








