"""Figure 02 generator for TRATLEQ manuscript.

Creates an 8-panel comparison of M158 (fall) and M181 (spring) sections for
Mip, Map, POC flux, and POC flux anomaly, masked below local bathymetry.
"""

import os
os.environ['PROJ_LIB'] = '/Users/joellehabib/anaconda3/pkgs/proj-8.2.1-hd69def0_0/share/proj'

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cmocean
from matplotlib.ticker import FuncFormatter
from pathlib import Path
import h5py


DATA_ROOT = Path("/Users/joellehabib/GIT/TRATLEQ/Data/DATA_PETER")
CRUISES_DIR = DATA_ROOT / "Cruises"


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
    """Return equally spaced contour levels for contour plots."""
    distance_levels = max_contour_level / levels
    contour_levels = np.arange(min_contour_level, max_contour_level, distance_levels)
    return contour_levels


def nonlinear_colormap():
    """Create the nonlinear colormap used for abundance/flux panels."""
    import pylab as pyl
    levels1 = [0, 1, 2]

    ####################################################################
    ###                      non linear colormap                     ###
    ####################################################################

    """
    nlcmap - a nonlinear cmap from specified levels

    Copyright (c) 2006-2007, Robert Hetland <hetland@tamu.edu>
    Release under MIT license.

    Some hacks added 2012 noted in code (@MRR)
    """

    from matplotlib.colors import LinearSegmentedColormap


    class nlcmap(LinearSegmentedColormap):
        """Nonlinear colormap.
           Needs the input of a linear colormap e.g. pylab.cm.jet
           and a list of levels e.g. [0,1,2,3,6]
           in order to change the interpolation, change the levels vector,
           e.g ad 20, 50 to make it more unlinear."""
        import numpy as np
        name = 'nlcmap'

        def __init__(self, cmap, levels):
            import numpy as np
            self.cmap = cmap
            # @MRR: Need to add N for backend
            self.N = cmap.N
            self.monochrome = self.cmap.monochrome
            self.levels = np.asarray(levels, dtype='float64')
            self._x = self.levels / self.levels.max()
            self._y = np.linspace(0.0, 1.0, len(self.levels))

        #@MRR Need to add **kw for 'bytes'
        def __call__(self, xi, alpha=1.0, **kw):
            import numpy as np
            """docstring for fname"""
            # @MRR: Appears broken?
            # It appears something's wrong with the
            # dimensionality of a calculation intermediate
            #yi = stineman_interp(xi, self._x, self._y)
            yi = np.interp(xi, self._x, self._y)
            return self.cmap(yi, alpha)


    cmap_nonlin = nlcmap(pyl.cm.CMRmap, levels1)
    return cmap_nonlin



cm1 = nonlinear_colormap()


os.chdir(DATA_ROOT)

# ------------------------------
# 1) Plot configuration and bathymetry
# ------------------------------

# Depth plotting range selector: set `n=0` for 0-6000 m, `n=1` for 0-1000 m
# Change `n` as needed before running the script
n = 0  # 0 -> 0-6000 m, 1 -> 0-1000 m
if n == 0:
    depth_limit = 6000
elif n == 1:
    depth_limit = 1000
else:
    depth_limit = 6000


# Legacy MAT bathymetry (fallback)
with h5py.File('topo_eq.mat', 'r') as mat_file:
    print(list(mat_file.keys()))

    depth_eq_mat = mat_file['depth_eq'][:]
    lon_eq_mat = mat_file['lon_eq'][:]

# Try loading updated topo CSVs for each cruise; fall back to MAT arrays
bottom_info_met_181 = pd.read_csv("bottom_info_met_181.csv", sep=',')
# Use the same bottom/topo CSV for M158 as for M181 (reuse 181 DataFrame)
bottom_info_met_158 = bottom_info_met_181.copy()

# Helper to extract lon/depth arrays robustly from a CSV
def _extract_topo(df):
    if {'depth_eq', 'lon_eq'}.issubset(df.columns):
        return df['depth_eq'].values, df['lon_eq'].values
    depth_col = next((c for c in df.columns if 'depth' in c.lower()), None)
    lon_col = next((c for c in df.columns if 'lon' in c.lower()), None)
    if depth_col and lon_col:
        return df[depth_col].values, df[lon_col].values
    return None, None


def load_interpolated_sections(cruise_id):
    """Load interpolated longitude, depth, flux, Mip, and Map arrays for one cruise."""
    os.chdir(CRUISES_DIR)
    lon = pd.read_csv(f"LAT_{cruise_id}_6000.csv", sep=',').values.squeeze()
    depth = np.abs(pd.read_csv(f"DEPTH_{cruise_id}_6000.csv", sep=',').values.squeeze())
    cflux = pd.read_csv(f"FluxC_{cruise_id}_6000.csv", sep=',').values
    mip = pd.read_csv(f"Mip_{cruise_id}_6000.csv", sep=',').values
    map_ = pd.read_csv(f"Map_{cruise_id}_6000.csv", sep=',').values
    return lon, depth, cflux, mip, map_


def apply_bottom_mask(map_, mip, cflux, depth, lon, depth_eq, lon_eq):
    """Mask section values below local bathymetry derived from nearest-topography longitude."""
    nearest_lon_indices = [np.argmin(np.abs(lon_eq - lon_val)) for lon_val in lon]
    nearest_depth_eq = depth_eq[nearest_lon_indices]

    map_masked = map_.copy()
    for col, depth_eq_val in enumerate(nearest_depth_eq):
        map_masked[:, col] = np.where(depth > depth_eq_val, np.nan, map_masked[:, col])

    mip_masked = np.where(np.isnan(map_masked), np.nan, mip)
    cflux_masked = np.where(np.isnan(map_masked), np.nan, cflux)
    return map_masked, mip_masked, cflux_masked


def compute_anomaly(field):
    """Remove the depth-wise mean to obtain longitudinal anomalies."""
    field_mean = np.nanmean(field, axis=1, keepdims=True)
    return field - field_mean

# Extract for 181, else use MAT
depth_eq_181, lon_eq_181 = _extract_topo(bottom_info_met_181)
if depth_eq_181 is None:
    depth_eq_181 = depth_eq_mat
    lon_eq_181 = lon_eq_mat

# Extract for 158, else use MAT
depth_eq_158, lon_eq_158 = _extract_topo(bottom_info_met_158)
if depth_eq_158 is None:
    depth_eq_158 = depth_eq_mat
    lon_eq_158 = lon_eq_mat




# ------------------------------
# 2) M158 sections (fall)
# ------------------------------

lon158, depth158, M158_Cflux, M158_Mip, M158_Map = load_interpolated_sections("158")
M158_Map, M158_Mip, M158_Cflux = apply_bottom_mask(
    M158_Map,
    M158_Mip,
    M158_Cflux,
    depth158,
    lon158,
    depth_eq_158,
    lon_eq_158,
)

west_end = -44.5


levels = 15
min_contour_level = 0
max_contour_level = 1
contour_levels = contour_levels_func(min_contour_level, max_contour_level, levels)
fig = plt.figure(1, figsize=(9,15))

ax = fig.add_subplot(423)
p1 = plt.contourf(lon158, depth158, M158_Map, contour_levels, cmap=cm1, alpha=1, extend='both')
ax.set_xlim([west_end, 5])
ax.tick_params(labelbottom=False)
ax.set_ylim([0, depth_limit])
ax.invert_yaxis()


ax.text(-0.1, 1, 'c)', transform=ax.transAxes, fontsize=7, fontweight='bold', va='top', ha='right')
ax.set_facecolor('tab:gray')



levels = 15
min_contour_level = 0
max_contour_level = 50

contour_levels = contour_levels_func(min_contour_level, max_contour_level, levels)
ax = fig.add_subplot(421)
p1 = plt.contourf(lon158, depth158, M158_Mip, contour_levels, cmap=cm1, alpha=1, extend='both')
ax.set_xlim([west_end, 5])
ax.tick_params(labelbottom=False)
ax.set_ylim([0, depth_limit])
ax.invert_yaxis()



ax.text(-0.1, 1, 'a)', transform=ax.transAxes, fontsize=7, fontweight='bold', va='top', ha='right')

plt.title('Fall: M158')

ax.set_facecolor('tab:gray')

levels = 15
min_contour_level = 0
max_contour_level = 30

contour_levels = contour_levels_func(min_contour_level, max_contour_level, levels)

ax = fig.add_subplot(425)
ax.set_xlim([west_end, 5])
ax.tick_params(labelbottom=False)
ax.set_ylim([0, depth_limit])
ax.invert_yaxis()
p1 = plt.contourf(lon158, depth158, M158_Cflux, contour_levels, cmap=cm1, alpha=1, extend='both')

ax.text(-0.1, 1, 'e)', transform=ax.transAxes, fontsize=7, fontweight='bold', va='top', ha='right')

ax.set_facecolor('tab:gray')

# POC flux anomaly (depth-wise mean removed)
New_value_C = compute_anomaly(M158_Cflux)

# POC flux anomaly panel
levels = 15
min_contour_level = -4
max_contour_level = 4
cm2 = cmocean.cm.balance

contour_levels = contour_levels_func(min_contour_level, max_contour_level, levels)

ax = fig.add_subplot(427)
ax.set_xlim([west_end, 5])
# show x-axis labels for panel (g)
ax.tick_params(labelbottom=True)
ax.xaxis.set_major_formatter(FuncFormatter(_lon_label))
ax.set_ylim([0, depth_limit])
ax.invert_yaxis()
p1 = plt.contourf(lon158, depth158, New_value_C, contour_levels, cmap=cm2, alpha=1, extend='both')
ax.text(-0.1, 1, 'g)', transform=ax.transAxes, fontsize=7, fontweight='bold', va='top', ha='right')

ax.set_facecolor('tab:gray')


# ------------------------------
# 3) M181 sections (spring)
# ------------------------------

lon181, depth181, M181_Cflux, M181_Mip, M181_Map = load_interpolated_sections("181")
M181_Map, M181_Mip, M181_Cflux = apply_bottom_mask(
    M181_Map,
    M181_Mip,
    M181_Cflux,
    depth181,
    lon181,
    depth_eq_181,
    lon_eq_181,
)


levels = 15
min_contour_level = 0
max_contour_level = 1
contour_levels = contour_levels_func(min_contour_level, max_contour_level, levels)


ax = fig.add_subplot(424)
p1 = plt.contourf(lon181, depth181, M181_Map, contour_levels, cmap=cm1, alpha=1, extend='both')
ax.set_xlim([west_end, 2])
ax.tick_params(labelbottom=False)
ax.set_ylim([0, depth_limit])
ax.invert_yaxis()
#colorbar
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.81, 0.51, 0.01, 0.18])
cbar = fig.colorbar(p1, cax=cbar_ax)

# Explicit ticks for Map colorbar and format to two decimals (avoid long floats)
ticks = np.linspace(0, 1.0, 6)
cbar.set_ticks(ticks)
cbar.set_ticklabels([f"{t:.2f}" for t in ticks])

cbar.ax.set_ylabel('Map \n (# L$^{-1}$)', fontsize=10)
ax.axes.yaxis.set_ticklabels([])
ax.text(-0.1, 1, 'd)', transform=ax.transAxes, fontsize=7, fontweight='bold', va='top', ha='right')

ax.set_facecolor('tab:gray')



levels = 15
min_contour_level = 0
max_contour_level = 50

contour_levels = contour_levels_func(min_contour_level, max_contour_level, levels)
ax = fig.add_subplot(422)
p1 = plt.contourf(lon181, depth181, M181_Mip, contour_levels, cmap=cm1, alpha=1, extend='both')
ax.set_xlim([west_end, 2])
ax.tick_params(labelbottom=False)
ax.set_ylim([0, depth_limit])
ax.invert_yaxis()
ax.axes.yaxis.set_ticklabels([])
plt.title('Spring: M181')

#colorbar
fig.subplots_adjust(right=0.8)

cbar_ax = fig.add_axes([0.81, 0.71, 0.01, 0.18])
cbar = fig.colorbar(p1, cax=cbar_ax)

cbar.ax.set_ylabel('Mip \n (# L$^{-1}$)', fontsize=10)



ax.text(-0.1, 1, 'b)', transform=ax.transAxes, fontsize=7, fontweight='bold', va='top', ha='right')
ax.set_facecolor('tab:gray')





levels = 15
min_contour_level = 0
max_contour_level = 30

contour_levels = contour_levels_func(min_contour_level, max_contour_level, levels)

ax = fig.add_subplot(426)
ax.set_xlim([west_end, 2])
ax.tick_params(labelbottom=False)
ax.set_ylim([0, depth_limit])
ax.invert_yaxis()
p1 = plt.contourf(lon181, depth181, M181_Cflux, contour_levels, cmap=cm1, alpha=1, extend='both')
ax.axes.yaxis.set_ticklabels([])

#colorbar
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.81, 0.3, 0.01, 0.18])
cbar = fig.colorbar(p1, cax=cbar_ax)
cbar.ax.set_ylabel('POC flux \n (mgC m$^{-2}$ day$^{-1}$)', fontsize=10)

ax.text(-0.1, 1, 'f)', transform=ax.transAxes, fontsize=7, fontweight='bold', va='top', ha='right')
ax.set_facecolor('tab:gray')

# POC flux anomaly (depth-wise mean removed)
New_value_C = compute_anomaly(M181_Cflux)

# anomaly of Cflux
levels = 15
min_contour_level = -4
max_contour_level = 4
cm2 = cmocean.cm.balance

contour_levels = contour_levels_func(min_contour_level, max_contour_level, levels)

ax = fig.add_subplot(428)
ax.set_xlim([west_end, 2])
# show x-axis labels for panel (h)
# enable bottom ticks and use longitude formatter like panel g
ax.tick_params(labelbottom=True)
ax.xaxis.set_major_formatter(FuncFormatter(_lon_label))
ax.set_ylim([0, depth_limit])
ax.invert_yaxis()
p1 = plt.contourf(lon181, depth181, New_value_C, contour_levels, cmap=cm2, alpha=1, extend='both')
ax.axes.yaxis.set_ticklabels([])
#colorbar
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.81, 0.1, 0.01, 0.18])
cbar = fig.colorbar(p1, cax=cbar_ax)
cbar.ax.set_ylabel('POC flux anomaly \n (mgC m$^{-2}$ day$^{-1}$)', fontsize=10)
ax.text(-0.1, 1, 'h)', transform=ax.transAxes, fontsize=7, fontweight='bold', va='top', ha='right')

ax.set_facecolor('tab:gray')

# Save figure to an explicit absolute path and report it
out_dir = Path("/Users/joellehabib/GIT/TRATLEQ/Plots/TRATLEQ_article/Journal1/DEC2025").expanduser()
out_dir.mkdir(parents=True, exist_ok=True)
# Choose filename based on depth-range selector `n`
fig_name = "Fig02.png" if n == 0 else "S4.png"
fig_path = out_dir / fig_name
#plt.savefig(fig_path, dpi=300, bbox_inches="tight")
print(f"Saved figure to: {fig_path}")
#plt.close()