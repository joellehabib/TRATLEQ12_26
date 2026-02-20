"""
Figure 01 generator for TRATLEQ manuscript.

This script builds an 8-panel figure combining:
1) Satellite SST Hovmöller sections,
2) Satellite Chl-a Hovmöller sections,
3) In situ CTD temperature sections (M158 and M181),
4) In situ bottle/HPLC Chl-a sections (2019 and 2022).
"""

import os
os.environ['PROJ_LIB'] = '/Users/joellehabib/anaconda3/pkgs/proj-8.2.1-hd69def0_0/share/proj'
from datetime import datetime, timedelta
from pathlib import Path

import cmocean
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.ticker import FuncFormatter
from scipy.interpolate import griddata
from seawater import dpth

DATA_ROOT = Path("/Users/joellehabib/GIT/TRATLEQ/Data/DATA_PETER")
CRUISES_DIR = DATA_ROOT / "Cruises"

cm = plt.get_cmap('inferno')

# ------------------------------
# Utility helpers
# ------------------------------


def matlab_datenum_to_datetime(value):
    """Convert one MATLAB datenum value to Python datetime."""
    return datetime.fromordinal(int(value)) + timedelta(days=value % 1) - timedelta(days=366)


def matlab_series_to_datetime(values):
    """Convert a vector of MATLAB datenum values to Python datetimes."""
    return [matlab_datenum_to_datetime(v) for v in values]


def format_time_axis(ax):
    """Apply monthly ticks and month/year labels to a time axis."""
    loc = mdates.MonthLocator(interval=1)
    ax.yaxis.set_major_locator(loc)
    fmt = mdates.DateFormatter('%b\n%Y')
    ax.yaxis.set_major_formatter(fmt)


def rasterize_contourf(contourf_obj):
    """Rasterize contourf collections to keep vector exports lightweight."""
    for coll in contourf_obj.collections:
        coll.set_rasterized(True)

    
def load_ctd_profiles(ctd_dir, include_cdom=False):
    """Load and concatenate CTD profile NetCDF files from one cruise directory."""
    os.chdir(ctd_dir)
    data_frames = []

    for filename in sorted(os.listdir()):
        if not filename.endswith('.nc'):
            continue

        print(f"Processing file: {filename}")
        dataset = xr.open_dataset(filename)
        ctd_value = filename.split('ctd_')[-1].split('.')[0]

        oxygen = np.squeeze(dataset.variables['DOX2'].values)
        latitude = np.repeat(dataset.variables['LATITUDE'].values[0], len(oxygen))
        longitude = np.repeat(dataset.variables['LONGITUDE'].values[0], len(oxygen))
        pressure = np.squeeze(dataset.variables['PRES'].values)
        time_var = np.repeat(dataset.variables['TIME'].values[0], len(oxygen))

        frame = {
            'Time': pd.to_datetime(time_var, unit='s', origin='unix'),
            'CTD': np.repeat(ctd_value, len(oxygen)),
            'Latitude': latitude,
            'Longitude': longitude,
            'PRES': pressure,
            'Oxygen': oxygen,
            'FLU2': np.squeeze(dataset.variables['FLU2'].values),
            'PAR': np.squeeze(dataset.variables['PAR'].values),
            'NOX': np.squeeze(dataset.variables['NOX'].values),
            'PSAL': np.squeeze(dataset.variables['PSAL'].values),
            'Temperature': np.squeeze(dataset.variables['TEMP'].values),
            'Turbidity': np.squeeze(dataset.variables['TURB'].values),
        }
        if include_cdom:
            frame['CDOM'] = np.squeeze(dataset.variables['CDOM'].values)

        data_frames.append(pd.DataFrame(frame))

    return pd.concat(data_frames, ignore_index=True)


def load_hplc_casts(filename, lat_col, lon_col):
    """Load and clean HPLC station logs for section gridding."""
    os.chdir(CRUISES_DIR)
    df_hplc = pd.read_excel(filename, engine="openpyxl")
    df_hplc = df_hplc.iloc[:-4]
    print(df_hplc.columns.tolist())

    df_hplc = df_hplc.dropna(subset=['Depth'])
    df_hplc = df_hplc[df_hplc['Tot_Chl_a'].notna()]

    latitude = df_hplc[lat_col]
    longitude = df_hplc[lon_col]
    pressure = df_hplc['Depth']
    return df_hplc, latitude, longitude, pressure


def _lon_label(x, pos=None):
    try:
        xx = float(x)
    except Exception:
        return ''
    if xx < 0:
        return f"{abs(int(xx))}W"
    elif xx > 0:
        return f"{int(xx)}E"
    else:
        return "0"

def contour_levels_func(min_contour_level, max_contour_level, levels):
    """Return equally spaced contour levels from min to max."""
    distance_levels = max_contour_level / levels
    contour_levels = np.arange(min_contour_level, max_contour_level, distance_levels)
    return contour_levels



def gridding_func(pos_min_max, depth_min_max, pos_array, depth, param):
    """Grid scattered section data using nearest-neighbour interpolation."""
    grid_method = "nearest"

    xi = np.linspace(min(pos_min_max), max(pos_min_max), 100)
    yi = np.linspace(min(depth_min_max), max(depth_min_max), 200)
    zi = griddata((pos_array, depth), param, (xi[None, :], yi[:, None]), method=grid_method)
    return xi, yi, zi


def gridding_func_gaussian(pos_min_max, depth_min_max, pos_array, depth, param,
                           h_influence=1.0, v_influence=3.0, h_cutoff=8.0, v_cutoff=20.0,
                           xi_res=200, yi_res=200):
    """
    Grid scattered observations with Gaussian weighting and rectangular cutoffs.

    Parameters  
    ----------
    pos_min_max : list
        [min, max] position (e.g., longitude) for output grid   
    depth_min_max : list
        [min, max] depth for output grid
    pos_array : array
        1D array of positions (e.g., longitude) of observations, here it will be the flux or the chl,...
    depth : array
        1D array of depths of observations
    param : array   
        1D array of parameter values of observations
    h_influence : float optional
        Horizontal influence scale (default is 1.0)  which means how smooth the output will be for horizontal
    v_influence : float optional    
        Vertical influence scale (default is 3.0)   which means how smooth the output will be for vertical
    h_cutoff : float optional
        Horizontal cutoff distance (default is 8.0) which means beyond this distance, no influence horizontally
    v_cutoff : float optional   
        Vertical cutoff distance (default is 20.0)  which means beyond this distance, no influence vertically
    xi_res : int optional   
        Number of output grid points in horizontal (default is 200)  which means how many points in the horizontal direction
    yi_res : int optional   
        Number of output grid points in vertical (default is 200)
    """
    import numpy as _np

    pos = _np.asarray(pos_array, dtype=_np.float32)
    dep = _np.asarray(depth, dtype=_np.float32)
    val = _np.asarray(param, dtype=_np.float32)

    xi = _np.linspace(min(pos_min_max), max(pos_min_max), xi_res, dtype=_np.float32)
    yi = _np.linspace(min(depth_min_max), max(depth_min_max), yi_res, dtype=_np.float32)

    X = xi[None, :]
    Y = yi[:, None]

    num = _np.zeros((yi_res, xi_res), dtype=_np.float32)
    den = _np.zeros_like(num)

    # Loop over observations to accumulate weighted sum (keeps memory use modest)
    for i in range(pos.size):
        dx = (X - pos[i]) / float(h_influence)
        dy = (Y - dep[i]) / float(v_influence)
        r2 = dx * dx + dy * dy

        # apply rectangular cutoff first to avoid computing full weight when far away
        mask = ( _np.abs(X - pos[i]) <= float(h_cutoff)) & ( _np.abs(Y - dep[i]) <= float(v_cutoff))
        if not mask.any():
            continue

        # intermediate Gaussian (between sharp and very smooth)
        w = _np.exp(-0.66 * r2, dtype=_np.float32)
        w = w * mask.astype(_np.float32)

        num += w * val[i]
        den += w

    with _np.errstate(invalid='ignore', divide='ignore'):
        zi = num / den

    # where denominator is zero, set NaN
    zi[den == 0] = _np.nan
    return xi, yi, zi

# ------------------------------
# Input data: trajectories and satellite products
# ------------------------------

os.chdir(CRUISES_DIR)

# CTD-derived Chl-a = 0.2 mg m^-3 contour used as overlay in panels g/h
df_contour = pd.read_csv('ctd_chl_0.2_contour_depth0-80m.csv')


os.chdir(DATA_ROOT)
# Float and cruise trajectories (position vs. time)
df_158 = pd.read_csv("traj_M158.csv", sep=',')
df_181 = pd.read_csv("traj_M181.csv", sep=',')
df_float22 = pd.read_csv("traj_float2022.csv", sep=',')

date_158 = matlab_series_to_datetime(df_158.Time.values)
date_181 = matlab_series_to_datetime(df_181.Time.values)
date_float22 = matlab_series_to_datetime(df_float22.Time.values)

# ------------------------------
# 1) Satellite SST (equatorial band average)
# ------------------------------

dataset = xr.open_dataset('OI_SST_TROPATL.nc')

# Read coordinates and variables
LAT_SST = dataset.variables['LAT'].values
LON_SST = dataset.variables['LON'].values
SST_TROPATL = np.squeeze(dataset.variables['SST_TROPATL'].values)
TIME_SST = dataset.variables['TIME'].values

mean_sst_tropatl = np.zeros((TIME_SST.size, LON_SST.size))

# Convert matlab datenum timestamps
date_sst = matlab_series_to_datetime(TIME_SST)

# Latitude subset used for zonal averaging (same choice as manuscript workflow)
SST_LAT_SLICE = slice(28, 35)

for k in range(TIME_SST.size):
    sst_tropatl = np.squeeze(SST_TROPATL[k, SST_LAT_SLICE, :])
    mean_sst_tropatl[k] = np.nanmean(sst_tropatl, axis=0)

# ------------------------------
# 2) Satellite chlorophyll-a (equatorial band average)
# ------------------------------
    
dataset = xr.open_dataset('CHLA_TROPATL.nc')

# Read coordinates and variables
LAT_SST = dataset.variables['LAT'].values
LON_CHL = dataset.variables['LON'].values
CHL_TROPATL = np.squeeze(dataset.variables['CHLA_TROPATL'].values)
TIME_SST = dataset.variables['TIME'].values

mean_chl_tropatl = np.zeros((TIME_SST.size, LON_CHL.size))

# Convert matlab datenum timestamps
date_chl = matlab_series_to_datetime(TIME_SST)

# Latitude subset used for zonal averaging around the equator (~1S–1N)
CHL_LAT_SLICE = slice(118, 123)

for k in range(TIME_SST.size):
    chl_tropatl = np.squeeze(CHL_TROPATL[k, CHL_LAT_SLICE, :])  # ~1S to 1N
    mean_chl_tropatl[k] = np.nanmean(chl_tropatl, axis=0)



jet = plt.get_cmap('Spectral')


# ------------------------------
# 3) Build multi-panel figure
# ------------------------------

# SST contour levels
levels = 100
min_contour_level =24
max_contour_level = 30
contour_levels = contour_levels_func(min_contour_level, max_contour_level, levels)

jet = plt.get_cmap('inferno')


# Panel (a): SST during M158 period
set_ylim_lower, set_ylim_upper = datetime(2019, 9, 1, 0, 0),datetime(2019, 11, 1, 0, 0)
fig = plt.figure(1, figsize=(7,8))
ax=fig.add_subplot(4,2,1)
p1 = plt.contourf(LON_SST, date_sst, mean_sst_tropatl,contour_levels, cmap=jet, alpha=1, extend='both')

plt.plot(df_158.Lon, date_158, 'bo',markersize=2)
plt.plot(df_181.Lon, date_181, 'bo',markersize=2)

ax.set_xlim([-40, 5.125])
ax.xaxis.set_major_formatter(FuncFormatter(_lon_label))
ax.set_ylim([set_ylim_lower, set_ylim_upper])
ax.text(-0.18, 1.15, '(a)', transform=ax.transAxes,fontsize=7, fontweight='bold', va='top', ha='right')

format_time_axis(ax)

plt.xticks(fontsize=7)
plt.yticks(fontsize=7)

ax.get_xaxis().set_ticklabels([])

plt.title('Fall: M158', fontsize=7)



rasterize_contourf(p1)


# Panel (b): SST during M181 period
levels = 100
min_contour_level =24
max_contour_level = 30
contour_levels = contour_levels_func(min_contour_level, max_contour_level, levels)

set_ylim_lower, set_ylim_upper = datetime(2022, 4, 1, 0, 0),datetime(2022, 6, 1, 0, 0)
ax=fig.add_subplot(4,2,2)
p1 = plt.contourf(LON_SST, date_sst, mean_sst_tropatl,contour_levels, cmap=jet, alpha=1,extend='both')

plt.plot(df_158.Lon, date_158, 'bo',markersize=2)
plt.plot(df_181.Lon, date_181, 'bo',markersize=2)

ax.set_xlim([-40, 5.125])
ax.xaxis.set_major_formatter(FuncFormatter(_lon_label))
ax.set_ylim([set_ylim_lower, set_ylim_upper])
ax.text(-0.1, 1.15, '(b)', transform=ax.transAxes,fontsize=7, fontweight='bold', va='top', ha='right')

format_time_axis(ax)
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)
ax.get_xaxis().set_ticklabels([])
plt.title('Spring: M181', fontsize=7)

#colorbar
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.81, 0.71, 0.01, 0.18])
cbar=fig.colorbar(p1, cax=cbar_ax)


cbar.ax.tick_params(labelsize=7) 
cbar.ax.locator_params(nbins=5)
cbar.ax.set_ylabel('SST (°)', fontsize=7)



rasterize_contourf(p1)

jet=cmocean.cm.algae







# Chl-a contour levels
levels = 20
min_contour_level = 0
max_contour_level =1
contour_levels = contour_levels_func(min_contour_level, max_contour_level, levels)

# Panel (c): Satellite Chl-a during M158 period
set_ylim_lower, set_ylim_upper = datetime(2019, 9, 1, 0, 0),datetime(2019, 11, 1, 0, 0)

ax=fig.add_subplot(4,2,3)
p1 = plt.contourf(LON_CHL, date_chl, mean_chl_tropatl,contour_levels, cmap=jet, alpha=1,vmin=0, vmax=1,extend='both')

plt.plot(df_158.Lon, date_158, 'bo',markersize=2)
plt.plot(df_181.Lon, date_181, 'bo',markersize=2)
plt.plot(df_float22.Lon, date_float22, color='blue')
ax.set_xlim([-40, 5.125])
ax.xaxis.set_major_formatter(FuncFormatter(_lon_label))
ax.set_ylim([set_ylim_lower, set_ylim_upper])
ax.text(-0.18, 1.15, '(c)', transform=ax.transAxes,fontsize=7, fontweight='bold', va='top', ha='right')

plt.xticks(fontsize=7)
plt.yticks(fontsize=7)

format_time_axis(ax)



# #colorbar
# cbar = plt.colorbar(p1)
# ax.set_ylabel('Time', fontsize=7)
# cbar.ax.locator_params(nbins=5)
# cbar.ax.tick_params(labelsize=7) 



rasterize_contourf(p1)

# Panel (d): Satellite Chl-a during M181 period

set_ylim_lower, set_ylim_upper = datetime(2022, 4, 1, 0, 0),datetime(2022, 6, 1, 0, 0)
ax=fig.add_subplot(424)
p1 = plt.contourf(LON_CHL, date_chl, mean_chl_tropatl,contour_levels, cmap=jet, alpha=1,vmin=0, vmax=1,extend='both')

plt.plot(df_158.Lon, date_158, 'bo',markersize=2)
plt.plot(df_181.Lon, date_181, 'bo',markersize=2)
plt.plot(df_float22.Lon, date_float22, color='blue')
ax.set_xlim([-40, 5.125])
ax.xaxis.set_major_formatter(FuncFormatter(_lon_label))
ax.set_ylim([set_ylim_lower, set_ylim_upper])
ax.text(-0.1, 1.10, '(d)', transform=ax.transAxes,fontsize=7, fontweight='bold', va='top', ha='right')

format_time_axis(ax)
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.81, 0.51, 0.01, 0.18])
cbar=fig.colorbar(p1, cax=cbar_ax)

#colorbar
# cbar = plt.colorbar(p1)
cbar.ax.set_ylabel('satellite Chl-a (mg m$^{-3}$)', fontsize=7)
cbar.ax.locator_params(nbins=5)
cbar.ax.tick_params(labelsize=7) 

rasterize_contourf(p1)

# ------------------------------
# 4) In situ CTD temperature section: M158 (panel e)
# ------------------------------

merged_df = load_ctd_profiles(CRUISES_DIR / "met_158_1_ctd", include_cdom=True)

# Diagnostic print for traceability
print(merged_df)
parameter_dict = {
    "Temperature": [12, 29],
}

depth_min_max = [0, 6000]
west_end = -43
transect_dict = {"0S": [-1, 1, [west_end, 5.125]]}
for transect in transect_dict:
    pos_min_max = transect_dict[transect][2]
    for parameter in parameter_dict:
        lat = np.array(merged_df['Latitude'])
        pos_array = np.array(merged_df['Longitude'])
        unique_pos = np.unique(pos_array)
        depth = np.array(dpth(np.array(merged_df["PRES"]), lat))

        param1 = np.array(merged_df['Temperature'])
        xi1, yi1, zi1 = gridding_func(pos_min_max, depth_min_max, pos_array, depth, param1)

        levels = 50
        min_contour_level = parameter_dict[parameter][0]
        max_contour_level = parameter_dict[parameter][1]
        contour_levels = contour_levels_func(min_contour_level, max_contour_level, levels)

        ax = fig.add_subplot(425)
        plt.rcParams.update({'font.size': 8})

        p1 = plt.contourf(
            xi1,
            yi1,
            zi1,
            contour_levels,
            cmap=cm,
            alpha=1,
            vmin=parameter_dict[parameter][0],
            vmax=parameter_dict[parameter][1],
        )
        contour = plt.contour(xi1, yi1, zi1, levels=[20, 25, 27], colors='black', linewidths=1)
        plt.xlim(-43, 2)
        ax.xaxis.set_major_formatter(FuncFormatter(_lon_label))
        plt.ylim(140, 0)
        ax.text(-0.18, 1.15, '(e)', transform=ax.transAxes, fontsize=7, fontweight='bold', va='top', ha='right')
        ax.set_ylabel('Depth (m)', fontsize=8)

        # Surface station markers
        plt.scatter(unique_pos, np.repeat(0, len(unique_pos)), c="k", marker="|", s=20)

        ax.get_xaxis().set_ticklabels([])
        plt.yticks(fontsize=8)
        ax.get_xaxis().set_ticklabels([])

 
alg=cmocean.cm.algae

# ------------------------------
# 5) In situ bottle/HPLC Chl-a section (2019, panel g)
# ------------------------------

df_, Latitude, Longitude, pressure = load_hplc_casts(
    "sep19atlStationlogHPLC_ODV.xlsx", lat_col='Latitude', lon_col='Longitude'
)

depth_min_max = [0, 1000]

transect_dict = {"0S": [-1, 1, [-43, 5.002]]}
parameter_dict = {'Tot_Chl_a': [0, 1]}

jet = plt.get_cmap('viridis')
for transect in transect_dict:
    pos_min_max = transect_dict[transect][2]

   
    for parameter in parameter_dict:
        lat = np.array(Latitude)
        pos_array = np.array(Longitude)
        unique_pos = np.unique(pos_array)
        depth = np.array(dpth(np.array(pressure),lat))
        param = np.array(df_[parameter])
            
            
        xi1, yi1, zi1 = gridding_func_gaussian(
            pos_min_max, depth_min_max, pos_array, depth, param,
            h_influence=1.0, v_influence=3.0, h_cutoff=8.0, v_cutoff=20.0,
            xi_res=300, yi_res=200
        )
        # Keep section depth to upper ocean for this panel
        zi1[yi1 > 150] = np.nan

        levels = 50
        min_contour_level = parameter_dict[parameter][0]
        max_contour_level = parameter_dict[parameter][1]
            
        contour_levels = contour_levels_func(min_contour_level, max_contour_level, levels)
            
        # Panel (g)
        
        ax = fig.add_subplot(4, 2, 7)
        plt.rcParams.update({'font.size': 8})  


        p2 = plt.contourf(xi1, yi1, zi1, contour_levels, cmap=alg, alpha=1, vmin=parameter_dict[parameter][0], vmax=parameter_dict[parameter][1])
        plt.scatter(unique_pos, np.repeat(0, len(unique_pos)), c="k", marker="|", s=20)

        ax.set_ylabel('Depth (m)', fontsize=8)
        plt.xlabel("Longitude (°)")
        plt.xlim(-43, 2)
        ax.xaxis.set_major_formatter(FuncFormatter(_lon_label))
        plt.ylim(100, 0)
        ax.text(-0.18, 1.15, '(g)', transform=ax.transAxes, fontsize=7, fontweight='bold', va='top', ha='right')
        plt.scatter(df_contour['Longitude'], df_contour['Depth'], c='k', s=1, label='CTD chl=0.2 contour (5-80m)')

      
       
        
# Colormap for in situ temperature sections
cm = plt.get_cmap('inferno')

# ------------------------------
# 6) In situ CTD temperature section: M181 (panel f)
# ------------------------------

parameter_dict = {
    "Temperature": [12, 29],
}

merged_df = load_ctd_profiles(CRUISES_DIR / "met_181_1_ctd", include_cdom=False)


depth_min_max = [0, 6000]
west_end = -43
transect_dict = {"0S": [-1, 1, [west_end, 5.125]]}
for transect in transect_dict:
    pos_min_max = transect_dict[transect][2]
    for parameter in parameter_dict:
        lat = np.array(merged_df['Latitude'])
        pos_array = np.array(merged_df['Longitude'])
        unique_pos = np.unique(pos_array)
        depth = np.array(dpth(np.array(merged_df["PRES"]), lat))

        param1 = np.array(merged_df['Temperature'])
        xi1, yi1, zi1 = gridding_func(pos_min_max, depth_min_max, pos_array, depth, param1)

        levels = 50
        min_contour_level = parameter_dict[parameter][0]
        max_contour_level = parameter_dict[parameter][1]
        contour_levels = contour_levels_func(min_contour_level, max_contour_level, levels)

        ax = fig.add_subplot(426)
        plt.rcParams.update({'font.size': 8})

        p1 = plt.contourf(
            xi1,
            yi1,
            zi1,
            contour_levels,
            cmap=cm,
            alpha=1,
            vmin=parameter_dict[parameter][0],
            vmax=parameter_dict[parameter][1],
        )
        contour = plt.contour(xi1, yi1, zi1, levels=[20, 25, 27], colors='black', linewidths=1)
        plt.xlim(-43, 2)
        ax.xaxis.set_major_formatter(FuncFormatter(_lon_label))
        plt.ylim(140, 0)
        ax.text(-0.1, 1.10, '(f)', transform=ax.transAxes, fontsize=7, fontweight='bold', va='top', ha='right')

        # Surface station markers
        plt.scatter(unique_pos, np.repeat(0, len(unique_pos)), c="k", marker="|", s=20)

        ax.get_xaxis().set_ticklabels([])

        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.81, 0.3, 0.01, 0.18])
        cb = fig.colorbar(p1, cax=cbar_ax)
        cb.ax.set_ylabel('T (°C)', fontsize=7)
        cb.ax.locator_params(nbins=5)
        cb.ax.tick_params(labelsize=7)

# ------------------------------
# 7) In situ bottle/HPLC Chl-a section (2022, panel h)
# ------------------------------

df_, Latitude, Longitude, pressure = load_hplc_casts(
    "apr22atlstationlogHPLC_CLASS20241118.xlsx", lat_col='lat', lon_col='lon'
)

depth_min_max = [0, 1000]

transect_dict = {"0S": [-1, 1, [-43, 5.002]]}
parameter_dict = {'Tot_Chl_a': [0, 1]}

jet = plt.get_cmap('viridis')
for transect in transect_dict:
    pos_min_max = transect_dict[transect][2]

   
    for parameter in parameter_dict:
        lat = np.array(Latitude)
        pos_array = np.array(Longitude)
        unique_pos = np.unique(pos_array)
        depth = np.array(dpth(np.array(pressure),lat))
        param = np.array(df_[parameter])
            
            
        xi1, yi1, zi1 = gridding_func_gaussian(
            pos_min_max, depth_min_max, pos_array, depth, param,
            h_influence=1.0, v_influence=3.0, h_cutoff=8.0, v_cutoff=20.0,
            xi_res=300, yi_res=200
        )
        # Keep section depth to upper ocean for this panel
        zi1[yi1 > 150] = np.nan

        levels = 50
        min_contour_level = parameter_dict[parameter][0]
        max_contour_level = parameter_dict[parameter][1]
            
        contour_levels = contour_levels_func(min_contour_level, max_contour_level, levels)
            
        # Panel (h)
        
        ax = fig.add_subplot(4, 2, 8)
        plt.rcParams.update({'font.size': 8})  


        p2 = plt.contourf(xi1, yi1, zi1, contour_levels, cmap=alg, alpha=1, vmin=parameter_dict[parameter][0], vmax=parameter_dict[parameter][1],extend='both')
        plt.scatter(unique_pos, np.repeat(0, len(unique_pos)), c="k", marker="|", s=20)

        plt.xlabel("Longitude (°)")
        plt.xlim(-43, 2)
        ax.xaxis.set_major_formatter(FuncFormatter(_lon_label))
        plt.ylim(100, 0)

        ax.text(-0.1, 1.10, '(h)', transform=ax.transAxes, fontsize=7, fontweight='bold', va='top', ha='right')

       
        
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.81, 0.1, 0.01, 0.18])
        cb = fig.colorbar(p2, cax=cbar_ax)
        
        cb.ax.locator_params(nbins=5)
        cb.ax.tick_params(labelsize=7)
        cb.ax.set_ylabel('In situ chl-a (mg m$^{-3}$)', fontsize=7)


# ------------------------------
# 8) Save output figure
# ------------------------------
out_dir = Path("/Users/joellehabib/GIT/TRATLEQ/Plots/TRATLEQ_article/Journal1/DEC2025").expanduser()
out_dir.mkdir(parents=True, exist_ok=True)
fig_path = out_dir / "Fig01.png"
#plt.savefig(fig_path, dpi=300, bbox_inches="tight")
print(f"Saved figure to: {fig_path}")
#plt.close()