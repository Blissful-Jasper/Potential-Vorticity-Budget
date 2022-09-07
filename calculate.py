# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 21:39:55 2022

@author: Administrator
"""

import numpy as np

import pandas as pd

import xarray as xr

import glob

from wrf import getvar,latlon_coords

from netCDF4 import Dataset

from metpy.units import units
import metpy.constants as constants
import metpy.calc as mpcalc


'''
This script is used to calculate the three terms on the right side of the equation: 
horizontal advection term, 
vertical advection term, 
non-adiabatic heating term
'''

####################################################################
#####     read data :wind-component ,longitude,latitude,time,level,pv,omega
####################################################################
# f_wind =  r"D:\PV\wrf-read\wrf-cal-pv\wind.nc"

f_wind =  r"/Users/xpji/wind.nc"
d_wind = xr.open_dataset(f_wind)
u = d_wind.u.data*units('m/s')                                      # units('m/s')
v = d_wind.v.data*units('m/s')                                       # units('m/s')


lat = d_wind.lat
lon = d_wind.lon
lev = d_wind.level  
time = d_wind.time                     

f_Q  =   r"/Users/xpji/Q.nc"
d_Q  =  xr.open_dataset(f_Q)

f_pv =  r"/Users/xpji/pv.nc"
pv   = xr.open_dataset(f_pv).pv

f_omega = r"/Users/xpji/omega.nc"
omega  = xr.open_dataset(f_omega).omega 
###################################################################################
######  calculate dx\dy\lon\lat
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

###################################################################################
######    Get the size of latitude and longitude
###################################################################################

path  = r'/Users/xpji/out0-8/'
filelist = glob.glob(path+'wrf*')
filelist.sort()
lon,lat,dx,dy = cal_dxdy(filelist[-1]) 
nx = lon.shape[0]
ny = lat.shape[0]
####################################################################
#####     calculate  the horizontal and vertical advection of PV
####################################################################

pv_adv = np.full((time.shape[0],lev.shape[0],lat.shape[0],lon.shape[0]),np.nan)

for i in range(time.shape[0]):
    
    for j in range(lev.shape[0]):
        
        print(i,j) 
        pv_adv[i,j,:,:] = mpcalc.advection(pv[i,j,:,:],u=u[i,j,:,:],v=v[i,j,:,:],dx=dx,dy=dy,x_dim=-1,y_dim=-2)
        

units.define('PVU = 1e-6 m^2 s^-1 K kg^-1 = pvu')
w = omega.data*units('Pa/s')
P = pv.data*units('PVU').to('m^2 s^-1 K kg^-1')
level = d_wind.level.data*units('hPa') 
wdPdp = np.zeros(shape=P.shape,dtype="float32")*(units("m^2 s^-1 K kg^-1")/units("hPa")*units('Pa/s'))
lev_dif = np.gradient(level)
p_dif   = np.gradient(P,axis=1)
for  i in range(40):
    
    for j in range(27):
        
        print(i,j)
    
        wdPdp[i,j,:,:] = w[i,j,:,:]*p_dif[i,j,:,:]/lev_dif[j]
pvvdv = wdPdp.to_base_units() 
pvvdv_m  = np.nanmean(pvvdv,axis=0)
pvvdv_a  =( pvvdv-pvvdv_m)
da_nc=xr.Dataset(
data_vars=dict(
    
    pvvdv=(['time','level','lat','lon'],-np.array(pvvdv.m)),
    pvvdv_ano=(['time','level','lat','lon'],-np.array(pvvdv_a.m)),
    ),      
                coords = {
                    'time':time,
                    'level':level.m,
                    'lat':lat,
                    'lon':lon,
                    
                    },
                attrs=dict(description="PV vertical advection(PVU = 1e-6 m^2 s^-1 K kg^-1)",
                            units= 'm^2 s^-2 K kg^-1',
                            equation='-<w*(dpv/dP)>-0907',
                            ),
                )
da_nc.to_netcdf(r'./data/pvvdv_.nc')  
###########################################################################################
Q    = d_Q.Q_to.data*units('K/s')    
####################################################################
#####     calculate gradient in 3 direction : x,y,p
####################################################################
# calculate  gradient
dv_dif = np.gradient(v,axis=1)
du_dif = np.gradient(u,axis=1)
dq_dif = np.gradient(Q,axis=1)

dv_dp = np.zeros(Q.shape,dtype='float32')*units('m/s')/units('hPa')
du_dp = np.zeros(Q.shape,dtype='float32')*units('m/s')/units('hPa')
dq_dp = np.zeros(Q.shape,dtype='float32')*units('K/s')/units('hPa')
ny=lat.shape[0]
fcc = mpcalc.coriolis_parameter(np.deg2rad(lat.data)).reshape((ny,1))

adv = np.zeros(Q.shape,dtype='float32')*units('K/s')/units('m')*units('m/s')/units('hPa')

dx = dx*units('m')
dy = dy*units('m')
q_dx = np.gradient(Q,axis=-1)
q_dy = np.gradient(Q,axis=-2)
for i in range(40):
    for j in range(27):
        print(i,j)
        dv_dp[i,j] = dv_dif[i,j]/lev_dif[j]
        du_dp[i,j] = du_dif[i,j]/lev_dif[j]
        dq_dp[i,j] = dq_dif[i,j]/lev_dif[j]

for i in range(40):
    for j in range(27):
        adv[i,j] = -1*dv_dp[i,j]*q_dx[i,j]/dx + du_dp[i,j]*q_dy[i,j]/dy

######### cal vertical components
vdv = np.zeros(Q.shape,dtype='float32')*units('K/s')/units('hPa')*units('m/s')/units('m')
v_dx = np.gradient(v, axis=-1)
u_dy = np.gradient(u, axis=-2)
vor_z = np.zeros(Q.shape,dtype='float32')*units('m/s')/units('m')

for i in range(40):
    for j in range(27):
        vor_z[i,j] = v_dx[i,j]/dx - u_dy[i,j]/dy

absvor_z = vor_z+fcc    

for i in range(40):
    for j in range(27):
        vdv[i,j] = absvor_z[i,j]*dq_dp[i,j]

heat = (adv+vdv)*constants.g
dia = heat.to_base_units()

dia_m = np.nanmean(dia,axis=0)
dia_ano = dia-dia_m

####################################################################
#####   output as Netcdf
####################################################################
heat_nc=xr.Dataset(
    data_vars=dict(
        
        Q_tot=(['time','level','lat','lon'],-np.array(dia.m)),
        Q_ana=(['time','level','lat','lon'],-np.array(dia_ano.m))),
     
        coords = {
                        'time':tim,
                        'level':lev.m,
                        'lat':lat.data,
                        'lon':lon.data,
                        
                        },
                    attrs=dict(description=" diabatic heating ",
                                units= 'm^2 s^-1 K kg^-1'),
                    )

heat_nc.to_netcdf(r'./data/diabatic_heat_20220907.nc')











