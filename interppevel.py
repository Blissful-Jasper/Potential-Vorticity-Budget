
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
This script is used to interpolate the data in wrf to the isobaric layer
'''
###################################################################################
######   get PV , interplevel to pressure
###################################################################################
def cal_pv_pv(file):
##########  先插值后计算PV  
##### getvar(ncfile,'pvo') 可以直接得到位涡
    ncfile = Dataset(file)
    pv = getvar(ncfile, "pvo")
    #### interp pv to pressure level
    pre = getvar(ncfile, "pressure")
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
    plevs =level
    pv_interp = np.array(interplevel(pv, pre, plevs))
    #### smooth   radius = 40km/12km=3.3km
    pv_inp = pv_interp
    return pv_inp


###################################################################################
######  get u,v,w,Q,
###################################################################################
def interplev(file):
    ncfile = Dataset(file)
    P = getvar(ncfile, "pressure")
    ua = getvar(ncfile, "ua")
    va = getvar(ncfile, "va")
    wa = getvar(ncfile, "wa")
    Q_mp = getvar(ncfile, "H_DIABATIC")
    Q_bl = getvar(ncfile, "RTHBLTEN")
    Q_ra = getvar(ncfile, "RTHRATEN")
    Q_to = Q_mp+Q_bl+Q_ra
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
    plevs =level
    u_interp = interplevel(ua, P, plevs)
    v_interp = interplevel(va, P, plevs)
    w_interp = interplevel(wa, P, plevs)
    Q_mp = interplevel(Q_mp,P,plevs)
    Q_bl = interplevel(Q_bl,P,plevs)
    Q_ra = interplevel(Q_ra,P,plevs)
    Q_to = interplevel(Q_to,P,plevs)
    return Q_mp,Q_bl,Q_ra,Q_to,u_interp,v_interp,w_interp

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

###################################################################################
######   Define the array:  wind speed component uvw, Q ,PV,
###################################################################################

u = np.full([len(filelist),27,ny,nx],np.nan,dtype='float32')
v = np.full([len(filelist),27,ny,nx],np.nan,dtype='float32')
w = np.full([len(filelist),27,ny,nx],np.nan,dtype='float32')
pv = np.full([len(filelist),27,ny,nx],np.nan,dtype='float32')
Q_mp = np.full([len(filelist),27,ny,nx],np.nan,dtype='float32')
Q_bl = np.full([len(filelist),27,ny,nx],np.nan,dtype='float32')
Q_ra = np.full([len(filelist),27,ny,nx],np.nan,dtype='float32')
Q_to = np.full([len(filelist),27,ny,nx],np.nan,dtype='float32')

for i in range(len(filelist)):


    pv[i,:,:,:]   =  cal_pv_pv(filelist[i])

    Q_mp[i,:,:,:],Q_bl[i,:,:,:],Q_ra[i,:,:,:],Q_to[i,:,:,:],u[i],v[i],w[i] =  interplev(filelist[i])


###################################################################################
######   The output is in netcdf format
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
        
        u=(['time','level','lat','lon'],u.data),
        v=(['time','level','lat','lon'],v.data),
        w=(['time','level','lat','lon'],w.data),
        ),      
                    coords = {
                        'time':time,
                        'level':lev.data,
                        'lat':lat.data,
                        'lon':lon.data,
                        
                        },
                    attrs=dict(description=" wind speed interpplevel ",
                               units= 'm/s'),
                    )
da_nc.to_netcdf(r'wind.nc')

da2_nc=xr.Dataset(
    data_vars=dict(
        
        Q_mp=(['time','level','lat','lon'],Q_mp.data),
        Q_bl=(['time','level','lat','lon'],Q_bl.data),
        Q_ra=(['time','level','lat','lon'],Q_ra.data),
        Q_to=(['time','level','lat','lon'],Q_to.data),
        ),      
                    coords = {
                        'time':time,
                        'level':lev.data,
                        'lat':lat.data,
                        'lon':lon.data,
                        
                        },
                    attrs=dict(description=" diabatic heating interpplevel  ",
                               units= 'K s-1'),
                    )
da2_nc.to_netcdf(r'Q.nc')
da3_nc=xr.Dataset(
    data_vars=dict(
        
        pv=(['time','level','lat','lon'],pv.data),
        ),      
                    coords = {
                        'time':time,
                        'level':lev.data,
                        'lat':lat.data,
                        'lon':lon.data,
                        
                        },
                    attrs=dict(description=" PV interpplevel  ",
                               units= 'K m**2 kg**-1 s**-1(PVU)'),
                    )
da3_nc.to_netcdf(r'pv.nc')
