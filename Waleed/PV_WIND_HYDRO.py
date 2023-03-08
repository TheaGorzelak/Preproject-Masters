# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 16:13:51 2022

@author: arsha
"""


from atlite.datasets.era5 import retrieve_data, _rename_and_clean_coords, retrieval_times, _area
from dask.utils import SerializableLock
import xarray as xr
import atlite
import logging
import cdsapi
from cartopy.io import shapereader
import geopandas as gpd
import pandas as pd
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LongitudeFormatter, LatitudeFormatter
import matplotlib.pyplot as plt
import matplotlib as mpl
logging.basicConfig(level=logging.INFO)

def natural_earth_shapes_EU(join_dict, drop_non_Europe=['MA','DZ','TN','GI']):
    # Download shape file (high resolution)
    shpfilename = shapereader.natural_earth(resolution='10m',
                                          category='cultural',
                                          name='admin_1_states_provinces')

    
    df =gpd.read_file(shpfilename)
    df = df.cx[-13:32,35:80]
    df = df[['iso_a2','geometry']]
    df = df.dissolve('iso_a2')
    df.index = list(df.index)
    drop_regions = drop_non_Europe
    # Absorbe microstates
    for main_r,sub_rs in join_dict.items():
        temp_main = df.loc[main_r,'geometry']
        for sub_r in sub_rs:
            drop_regions.append(sub_r)
            temp_r = df.loc[sub_r,'geometry']
        
            temp_main = temp_main.union(temp_r)
        temp_main = gpd.GeoSeries([temp_main])
        df.loc[[main_r],'geometry'] = temp_main.values
    
    df = df.drop(index=drop_regions)
    return df

# Create Europe shapefile
join_dict = {'FR':['GG','AD','MC'],'IT':['VA','SM'], 'GB':['JE','IM'],'FI':['AX'],'NO':['FO'],
            'CH':['LI'], 'BE':['LU'],'RS':['XK']}
europe = natural_earth_shapes_EU(join_dict)


# ___________________________#

esgf_params = {
   'data_node': 'esg-dn2.nsc.liu.se',
   'source_id': 'EC-Earth3',
   'experiment_id': 'historical',
   'project' : 'CMIP6',
   'frequency':'3hr'
}

cutout_ssp585_cmip_Earth3_20002014 = atlite.Cutout(path='D:\cmip\cmip_europe_EC-Earth3_20002014_historical_r1i1p1f1.nc',
                            module=['cmip'],                 
                            x=slice(-13,45),
                            y=slice(32,83),
                            time=slice("2000-01-01","2014-12-31"),
                            esgf_params=esgf_params,
                            dt='3H',dx=1, dy=1)

cutout_ssp585_cmip_Earth3_20002014.prepare()

esgf_params = {
   'data_node': 'esg-dn2.nsc.liu.se',
   'source_id': 'EC-Earth3',
   'experiment_id': 'ssp126',
   'project' : 'CMIP6',
   'frequency':'3hr'
}

cutout_ssp585_cmip_Earth3_20152030 = atlite.Cutout(path='D:\cmip\cmip_europe_EC-Earth3_20152030_585_r1i1p1f1.nc',
                            module=['cmip'],                 
                            x=slice(-13,45),
                            y=slice(32,83),
                            time=slice("2015-01-01","2015-12-31"),
                            esgf_params=esgf_params,
                            dt='3H',dx=1, dy=1)

cutout_ssp585_cmip_Earth3_20152030.prepare()

esgf_params = {
   'data_node': 'esg-dn2.nsc.liu.se',
   'source_id': 'EC-Earth3',
   'experiment_id': 'ssp585',
   'project' : 'CMIP6',
   'frequency':'3hr'
}

cutout_ssp585_cmip_Earth3_20702100 = atlite.Cutout(path='D:\cmip\cmip_europe_EC-Earth3_20702100_585_r1i1p1f1.nc',
                            module=['cmip'],                 
                            x=slice(-13,45),
                            y=slice(32,83),
                            time=slice("2070-01-01","2100-01-01"),
                            esgf_params=esgf_params,
                            dt='3H',dx=1, dy=1)

cutout_ssp585_cmip_Earth3_20702100.prepare()
# ___________________________#


# Wind Capacity factor CMIP
# The surface roughness is not available from the CMIP database so ERA5 roughness is used instead.|
import cdsapi

c = cdsapi.Client()

c.retrieve(
    'reanalysis-era5-single-levels-monthly-means',
    {
        'format': 'netcdf',
        'product_type': 'monthly_averaged_reanalysis',
        'variable': 'forecast_surface_roughness',
        'year': '2019',
        'month': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
        ],
        'area': [
            83, -13, 32,
            45,
        ],
        'time': '00:00',
    },
    'roughness.nc')

from atlite.datasets.era5 import _rename_and_clean_coords

# Create roughness data based on 1 year average from ERA5. Then interpolate the ERA5 data into the resolution of CMIP.
roughness = xr.open_dataset('roughness.nc')

roughness = roughness.rename({'fsr':'roughness'})

roughness = roughness.mean(dim='time')

roughness = _rename_and_clean_coords(roughness)
roughness.roughness.attrs['prepared_feature'] = 'wind'

da = roughness.roughness.interp_like(cutout_ssp585_cmip_Earth3_20702100.data['influx'].isel(time=0))

cutout_ssp585_cmip_Earth3_20002014.data = cutout_ssp585_cmip_Earth3_20002014.data.assign(roughness=da)
cutout_ssp585_cmip_Earth3_20152030.data = cutout_ssp585_cmip_Earth3_20152030.data.assign(roughness=da)
cutout_ssp585_cmip_Earth3_20702100.data = cutout_ssp585_cmip_Earth3_20702100.data.assign(roughness=da)

# ___________________________#

wind_ssp585_cp_Earth3_20002014 = cutout_ssp585_cmip_Earth3_20002014.wind('Vestas_V112_3MW', shapes=europe,capacity_factor=True, per_unit=True)

wind_ssp585_cp_Earth3_20152030 = cutout_ssp585_cmip_Earth3_20152030.wind('Vestas_V112_3MW', shapes=europe,capacity_factor=True, per_unit=True)

wind_ssp585_cp_Earth3_BOC_585 = xr.concat([wind_ssp585_cp_Earth3_20002014, wind_ssp585_cp_Earth3_20152030], dim="time")

wind_ssp585_cp_Earth3_EOC_585 = cutout_ssp585_cmip_Earth3_20702100.wind('Vestas_V112_3MW', shapes=europe,capacity_factor=True, per_unit=True)


cf_ssp585_wind_Earth3_BOC_585 = wind_ssp585_cp_Earth3_BOC_585.to_series().unstack()['2000-01-01':'2029-12-31'].resample('D').mean().fillna(0)

cf_ssp585_wind_Earth3_EOC_585 = wind_ssp585_cp_Earth3_EOC_585.to_series().unstack()['2070-01-01':'2099-12-31'].resample('D').mean().fillna(0)


# Drop 29th Feb

cf_ssp585_wind_Earth3_BOC_585 = cf_ssp585_wind_Earth3_BOC_585[~((cf_ssp585_wind_Earth3_BOC_585.index.month == 2) & (cf_ssp585_wind_Earth3_BOC_585.index.day == 29))]

cf_ssp585_wind_Earth3_EOC_585 = cf_ssp585_wind_Earth3_EOC_585[~((cf_ssp585_wind_Earth3_EOC_585.index.month == 2) & (cf_ssp585_wind_Earth3_EOC_585.index.day == 29))]


# cf_ssp585_wind_ensemble_BOC = (cf_ssp585_wind_ESM2_BOC_585+cf_ssp585_wind_CM2_BOC_585+cf_ssp585_wind_Earth3_BOC_585)/3

# wind_ensemble_BOC_ssp585 = cf_ssp585_wind_ensemble_BOC.groupby([cf_ssp585_wind_ensemble_BOC.index.month, cf_ssp585_wind_ensemble_BOC.index.day]).mean()

# cf_ssp585_wind_ensemble_EOC = (cf_ssp585_wind_ESM2_EOC_585+cf_ssp585_wind_CM2_EOC_585+cf_ssp585_wind_Earth3_EOC_585)/3

# wind_ensemble_EOC_ssp585 = cf_ssp585_wind_ensemble_EOC.groupby([cf_ssp585_wind_ensemble_EOC.index.month, cf_ssp585_wind_ensemble_EOC.index.day]).mean()

# wind_ensemble_BOC_ssp585.to_csv('wind_ensemble_BOC_ssp585.csv', index=True)

# wind_ensemble_EOC_ssp585.to_csv('wind_ensemble_EOC_ssp585.csv', index=True)


# Hydro

def hydropowerplants(hydrotype,hydropower_database,datapath,countries):
    import pandas as pd
    import geopandas as gpd
    
    # Read hydropower plant database
    h = pd.read_csv(hydropower_database,index_col = [5])
    h['country_code'] = h.index
    h['index'] = np.arange(len(h))  
    h = h.loc[countries]
    h.set_index('index',inplace=True)   
    h['Coordinates'] = list(zip(h.lon, h.lat))
    h['Coordinates'] = h['Coordinates'].apply(Point)
    
    # Remove hydropower plants which causes error in the ATLITE conversion
    arrays_drop = [0]*len(countries)
    for c in range(len(countries)):
        ad = pd.read_csv(datapath + 'arr_drop_' + countries[c] + '_all.csv')
        ad_w = np.array(ad['0'])
        if countries[c] == 'ES':
            ad2 = np.array(pd.read_csv(datapath + 'Spain_debug_array.csv'))
            ad2 = ad2[ad2!=0]
            ad_w = np.concatenate([ad_w,ad2])
        elif countries[c] == 'PT':
            ad2 = np.array(pd.read_csv(datapath + 'Portugal_debug_array.csv'))
            ad2 = ad2[ad2!=0]
            ad_w = np.concatenate([ad_w,ad2])

        arrays_drop[c] = ad_w
    array_drop_big = np.sort(np.concatenate(arrays_drop).astype(int))
    h_big = h.drop(array_drop_big)
    
    # Remove hydropower plants which are not located within the country borders
    latlonrange = pd.read_csv(datapath + 'lat_lon_rang.csv',index_col = 0,sep=';') # Latitude and longitude country boundaries 
    h_big_drop = [0]*len(countries)
    for c in range(len(countries)):
        country = countries[c]
        h_big_c = h_big[h_big['country_code'] == country]
        h_big_drop[c] = h_big_c.drop(h_big_c[h_big_c.lat < latlonrange.loc[country].maxlatitude][h_big_c.lat > latlonrange.loc[country].minlatitude][h_big_c.lon < latlonrange.loc[country].maxlongitude][h_big_c.lon > latlonrange.loc[country].minlongitude].index).index
    h_big_drop_indices = np.concatenate(h_big_drop) # Outside of country border
    array_drop_big_2 = np.sort(np.concatenate([array_drop_big,h_big_drop_indices]))
    h_big = h.drop(array_drop_big_2)
    if hydrotype == 'all':
        hplants = gpd.GeoDataFrame(h_big, geometry='Coordinates')
    else:
        hplants = gpd.GeoDataFrame(h_big[h_big.type == hydrotype], geometry='Coordinates')

    return hplants

import numpy as np
import atlite
from shapely.geometry import Point

#%% ============================= INPUT =======================================

# Type of hydropower plants included:
hydrotype = 'HDAM' # all/HDAM/HPHS/HROR

# Cutout directory:
cutout_dir = "cutouts"

# General data path:
gendatapath = 'gendata/'

# Results data path:
resdatapath = 'resdata/'

# Countries:
country_codes = ['SE','ES','NO','AT','BG','FI','HR','ME','PT','RO','CH','FR','IT','DE','CZ','HU','PL','SK','BA','MK','RS','SI']

# country_codes = ['AL','AT','BA','BE','BG','BY','CH','CZ','DE','DK','EE','ES','FI','FR','GB','GR','HR','HU','IE','IT','LT','LV','MD','ME','MK','MT','NL','NO','PL','PT','RO','RS','RU','SE','SI','SK','TR','UA']

#%% ============================= OUTPUT ======================================s
hydropower_database = gendatapath + 'hydro-power-database-master/data/jrc-hydro-power-plant-database.csv'
hydrobasin_database = gendatapath + 'hydroBASINS/hybas_eu_lev08_v1c.shp'
indices = pd.read_csv(gendatapath + 'index.csv',sep=';',index_col=[0,1,5])
#array_drop_big = pd.read_csv(datapath + 'arr_drop_big.csv').values # Array containing indices of all hydro power plant which is inapplicable in the ATLITE conversion due to geometric conditions 
hplants = hydropowerplants(hydrotype,hydropower_database,gendatapath,country_codes)


hydro_ssp585_cp_Earth3_20002014 = cutout_ssp585_cmip_Earth3_20002014.hydro(hplants,hydrobasin_database, capacity_factor=True, per_unit=True)

hydro_ssp585_cp_Earth3_20152030 = cutout_ssp585_cmip_Earth3_20152030.hydro(hplants,hydrobasin_database, capacity_factor=True, per_unit=True)

hydro_ssp585_cp_Earth3_BOC_585 = xr.concat([hydro_ssp585_cp_Earth3_20002014, hydro_ssp585_cp_Earth3_20152030], dim="time")

hydro_ssp585_cp_Earth3_EOC_585 = cutout_ssp585_cmip_Earth3_20702100.hydro(hplants,hydrobasin_database, capacity_factor=True, per_unit=True)


# ___________________________#
cf_hydro_Earth3_BOC_585 = hydro_ssp585_cp_Earth3_BOC_585.to_series().unstack()

cf_hydro_Earth3_EOC_585 = hydro_ssp585_cp_Earth3_EOC_585.to_series().unstack()

# ___________________________#
#Earth3
cf_hydro_Earth3_BOC_585_df = pd.DataFrame(
                     index=pd.Series(
                     data = cf_ssp585_pv_Earth3_BOC_585.index,
                     name = 'utc_time'))

for c in range(len(country_codes)):
                    c_index = hplants[hplants['country_code'] == country_codes[c]].index
                    inflow_agg = cf_hydro_Earth3_BOC_585.loc[c_index].sum(axis=0)
                    cf_hydro_Earth3_BOC_585_df[c]=inflow_agg
                    
cf_hydro_Earth3_EOC_585_df = pd.DataFrame(
                     index=pd.Series(
                     data = cf_ssp585_pv_Earth3_EOC_585.index,
                     name = 'utc_time'))


for c in range(len(country_codes)):
                    c_index = hplants[hplants['country_code'] == country_codes[c]].index
                    inflow_agg = cf_hydro_Earth3_EOC_585.loc[c_index].sum(axis=0)
                    cf_hydro_Earth3_EOC_585_df[c]=inflow_agg 