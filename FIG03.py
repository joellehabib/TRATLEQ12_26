"""Figure 03 generator for TRATLEQ manuscript.

This script generates a figure comparing M158 (fall) and M181
(spring) for:
- Mip vertical sections (top row),
- LADCP zonal velocity (UCUR) sections (bottom row),
with the maximum EUC core depth/longitude overlaid.
"""

import os
from pathlib import Path

import cmocean
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.ticker import FuncFormatter
from scipy.interpolate import griddata
from seawater import dpth

os.environ["PROJ_LIB"] = (
    "/Users/joellehabib/anaconda3/pkgs/proj-8.2.1-hd69def0_0/share/proj"
)

DATA_ROOT = Path("/Users/joellehabib/GIT/TRATLEQ/Data/DATA_PETER")
CRUISES_DIR = DATA_ROOT / "Cruises"
M158_LADCP_DIR = Path("/Users/joellehabib/GIT/TRATLEQ/Data/Campaigns/ADCP/M158/LADCP/nc")
M181_LADCP_DIR = Path("/Users/joellehabib/GIT/TRATLEQ/Data/Campaigns/ADCP/M181/LADCP/nc")
OUT_DIR = Path(
    "/Users/joellehabib/GIT/TRATLEQ/Plots/TRATLEQ_article/Journal1/DEC2025"
)

WEST_END = -45
EAST_END = -10
DEPTH_MIN_MAX_U = [0, -1000]
UCUR_DEPTH_PLOT_RANGE = (-300, 0)
MIP_DEPTH_PLOT_RANGE = (-300, 0)

CM_UCUR = cmocean.cm.balance
CM_MIP = None  # assigned in main via nonlinear_colormap()


def contour_levels_func(min_contour_level, max_contour_level, levels):
    """Return equally spaced contour levels for contour plots."""
    distance_levels = max_contour_level / levels
    return np.arange(min_contour_level, max_contour_level, distance_levels)


def gridding_func(pos_min_max, depth_min_max, pos_array, depth, param):
    """Interpolate scattered observations onto a regular lon-depth grid."""
    xi = np.linspace(min(pos_min_max), max(pos_min_max), 1000)
    yi = np.linspace(min(depth_min_max), max(depth_min_max), 200)
    zi = griddata((pos_array, depth), param, (xi[None, :], yi[:, None]), method="linear")
    return xi, yi, zi


def nonlinear_colormap():
    """Create the nonlinear colormap used for Mip sections."""
    import pylab as pyl
    from matplotlib.colors import LinearSegmentedColormap

    levels1 = [0, 1, 2]

    class NonLinearCmap(LinearSegmentedColormap):
        name = "nlcmap"

        def __init__(self, cmap, levels):
            self.cmap = cmap
            self.N = cmap.N
            self.monochrome = self.cmap.monochrome
            self.levels = np.asarray(levels, dtype="float64")
            self._x = self.levels / self.levels.max()
            self._y = np.linspace(0.0, 1.0, len(self.levels))

        def __call__(self, xi, alpha=1.0, **kw):
            yi = np.interp(xi, self._x, self._y)
            return self.cmap(yi, alpha)

    return NonLinearCmap(pyl.cm.CMRmap, levels1)


def _lon_label(x, pos=None):
    try:
        xx = float(x)
    except Exception:
        return ""
    if xx < 0:
        return f"{abs(int(xx))}W"
    if xx > 0:
        return f"{int(xx)}E"
    return "0"


def _depth_label(y, pos=None):
    try:
        return f"{int(abs(y))}"
    except Exception:
        return ""


def load_ladcp_profiles(ladcp_dir):
    """Load all LADCP netCDF profiles in one cruise folder into one DataFrame."""
    os.chdir(ladcp_dir)
    frames = []

    for filename in sorted(os.listdir()):
        if not filename.endswith(".nc"):
            continue

        print(f"Processing file: {filename}")
        dataset = xr.open_dataset(filename)
        ctd_value = filename.split("ladcp_")[-1].split(".")[0]

        ucur = np.squeeze(dataset.variables["UCUR"].values)
        latitude = np.repeat(dataset.variables["LATITUDE"].values[0], len(ucur))
        longitude = np.repeat(dataset.variables["LONGITUDE"].values[0], len(ucur))
        depth = np.squeeze(dataset.variables["DEPTH"].values)
        time_var = np.repeat(dataset.variables["TIME"].values[0], len(ucur))

        frames.append(
            pd.DataFrame(
                {
                    "Time": pd.to_datetime(time_var, unit="s", origin="unix"),
                    "CTD": np.repeat(ctd_value, len(ucur)),
                    "Latitude": latitude,
                    "Longitude": longitude,
                    "PRES": depth,
                    "UCUR": ucur,
                    "VCUR": np.squeeze(dataset.variables["VCUR"].values),
                    "ECUR": np.squeeze(dataset.variables["ECUR"].values),
                }
            )
        )

    return pd.concat(frames, ignore_index=True)


def compute_euc_core(df_ladcp, max_depth=200, lon_min=-39):
    """Compute maximum UCUR location per CTD (EUC core proxy)."""
    df_work = df_ladcp.copy()
    df_work.loc[df_work["PRES"] > max_depth, "UCUR"] = np.nan

    max_euc = df_work.loc[df_work.groupby("CTD")["UCUR"].idxmax()].reset_index(drop=True)
    max_euc.loc[max_euc["Longitude"] < lon_min, :] = np.nan
    return max_euc


def load_mip_section(cruise_id):
    """Load interpolated Mip section for one cruise at 0-300 m products."""
    os.chdir(CRUISES_DIR)
    lon = pd.read_csv(f"LAT_{cruise_id}_300.csv", sep=",").values.squeeze()
    depth = pd.read_csv(f"DEPTH_{cruise_id}_300.csv", sep=",").values.squeeze()
    mip = pd.read_csv(f"Mip_{cruise_id}_300.csv", sep=",").values
    return lon, depth, mip


def plot_mip_panel(ax, lon, depth, mip, max_euc, panel_label, title=None, hide_ylabels=False):
    """Render one Mip panel."""
    contour_levels = contour_levels_func(10, 80, 50)
    p = ax.contourf(lon, depth, mip, contour_levels, cmap=CM_MIP, alpha=1, extend="both")

    ax.set_xlim([WEST_END, EAST_END])
    ax.set_ylim(MIP_DEPTH_PLOT_RANGE)
    ax.yaxis.set_major_formatter(FuncFormatter(_depth_label))
    ax.tick_params(axis="y", labelsize=8)
    if hide_ylabels:
        ax.get_yaxis().set_ticklabels([])
    ax.text(-0.1, 1.15, panel_label, transform=ax.transAxes, fontsize=7, fontweight="bold", va="top", ha="right")

    if title:
        ax.set_title(title)

    ax.plot(max_euc.Longitude, -max_euc.PRES, color="grey")
    ax.get_xaxis().set_ticklabels([])
    return p


def plot_ucur_panel(ax, df_ladcp, max_euc, panel_label, hide_ylabels=False):
    """Render one UCUR panel from gridded LADCP section."""
    lat = np.array(df_ladcp["Latitude"])
    pos_array = np.array(df_ladcp["Longitude"])
    depth = np.array(dpth(np.array(df_ladcp["PRES"]), lat) * -1)
    param = np.array(df_ladcp["UCUR"])

    xi, yi, zi = gridding_func([WEST_END, 5.125], DEPTH_MIN_MAX_U, pos_array, depth, param)
    contour_levels = contour_levels_func(-1.2, 1.2, 15)

    p = ax.contourf(xi, yi, zi, contour_levels, cmap=CM_UCUR, alpha=1, extend="both")
    ax.plot(max_euc.Longitude, -max_euc.PRES, color="grey")
    ax.set_xlabel("Longitude (°)")
    ax.xaxis.set_major_formatter(FuncFormatter(_lon_label))
    ax.tick_params(axis="x", labelsize=8)
    ax.set_ylim(UCUR_DEPTH_PLOT_RANGE)
    ax.yaxis.set_major_formatter(FuncFormatter(_depth_label))
    ax.tick_params(axis="y", labelsize=8)
    ax.set_xlim([WEST_END, EAST_END])
    if hide_ylabels:
        ax.tick_params(labelleft=False)
    ax.text(-0.1, 1.10, panel_label, transform=ax.transAxes, fontsize=7, fontweight="bold", va="top", ha="right")
    return p


def build_figure():
    """Create the full 2x2 figure and save it."""
    global CM_MIP
    CM_MIP = nonlinear_colormap()

    # Load cruises
    m158_ladcp = load_ladcp_profiles(M158_LADCP_DIR)
    m181_ladcp = load_ladcp_profiles(M181_LADCP_DIR)

    # EUC core lines
    m158_euc = compute_euc_core(m158_ladcp)
    m181_euc = compute_euc_core(m181_ladcp)

    # Mip sections
    lon158, depth158, mip158 = load_mip_section("158")
    lon181, depth181, mip181 = load_mip_section("181")

    fig = plt.figure(1, figsize=(10, 5))

    # Top row: Mip
    ax_a = fig.add_subplot(221)
    plot_mip_panel(ax_a, lon158, depth158, mip158, m158_euc, "(a)", title="Fall: M158")

    ax_b = fig.add_subplot(222)
    p_mip181 = plot_mip_panel(ax_b, lon181, depth181, mip181, m181_euc, "(b)", title="Spring: M181", hide_ylabels=True)

    # Bottom row: UCUR
    ax_c = fig.add_subplot(223)
    plot_ucur_panel(ax_c, m158_ladcp, m158_euc, "(c)")

    ax_d = fig.add_subplot(224)
    p_ucur181 = plot_ucur_panel(ax_d, m181_ladcp, m181_euc, "(d)", hide_ylabels=True)

    # Colorbars
    fig.subplots_adjust(right=0.8)

    cbar_ax_top = fig.add_axes([0.82, 0.5, 0.01, 0.38])
    cbar_top = fig.colorbar(p_mip181, cax=cbar_ax_top)
    cbar_top.ax.tick_params(labelsize=7)
    cbar_top.ax.locator_params(nbins=5)
    cbar_top.ax.set_ylabel("Mip \n (# L$^{-1}$)", fontsize=10)

    cbar_ax_bottom = fig.add_axes([0.82, 0.1, 0.01, 0.38])
    cbar_bottom = fig.colorbar(p_ucur181, cax=cbar_ax_bottom)
    cbar_bottom.ax.tick_params(labelsize=7)
    cbar_bottom.ax.locator_params(nbins=5)
    cbar_bottom.ax.set_ylabel("Zonal velocity \n (m s$^{-1}$)", fontsize=10)

    # Shared y-label for left column
    fig.text(0.055, 0.5, "Depth (m)", va="center", ha="center", rotation="vertical", fontsize=10)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_path = OUT_DIR / "Fig03.png"
    #plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure to: {fig_path}")
    #plt.close()


if __name__ == "__main__":
    build_figure()
