"""Figure 04 for TRATLEQ manuscript.

This script generates two-panel map figure (Fall and Spring) combining:
- GLORYS-derived upper-ocean current magnitude and vectors,
- in situ MiP abundance points from EUC campaigns,
- coastline, rivers, and bathymetry contours.

"""

import os
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import gsw
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
from scipy.interpolate import griddata

os.environ["PROJ_LIB"] = (
    "/Users/joellehabib/anaconda3/pkgs/proj-8.2.1-hd69def0_0/share/proj"
)

DATA_DIR = Path("/Users/joellehabib/GIT/TRATLEQ/Data")
EUC_DIR = DATA_DIR / "Campaigns/EUC_withoutextrap"
RIVERS_GDB_DIR = DATA_DIR / "Campaigns/HydroRIVERS_v10_sa.gdb"
SAT_FALL_DIR = DATA_DIR / "Campaigns/Satellite_data/Fall_2019"
SAT_SPRING_DIR = DATA_DIR / "Campaigns/Satellite_data/Spring_2018_2023"
OUT_DIR = Path("/Users/joellehabib/GIT/TRATLEQ/Plots/TRATLEQ_article/Journal1/DEC2025")

MAP_EXTENT = (-55, -30, -10, 10)
DENSITY_MIN = 24.5
DENSITY_MAX = 26.3


def get_real_season(date):
    """Return climatological season for a pandas timestamp."""
    month = date.month
    day = date.day

    if (month == 12 and day >= 21) or (month <= 3 and (month < 3 or day <= 20)):
        return "Winter"
    if (month == 3 and day >= 21) or (month < 6 or (month == 6 and day < 21)):
        return "Spring"
    if (month == 6 and day >= 21) or (month < 9 or (month == 9 and day < 21)):
        return "Summer"
    return "Fall"


def load_bathymetry_grid(bathy_csv, extent, resolution=200):
    """Load and interpolate Atlantic bathymetry onto a regular lon-lat grid."""
    bathymetry_data = pd.read_csv(bathy_csv, sep=",")
    lon_bath = bathymetry_data["x"]
    lat_bath = bathymetry_data["y"]
    depth_bath = bathymetry_data["Depth"]

    lon_min, lon_max, lat_min, lat_max = extent
    grid_lon = np.linspace(lon_min, lon_max, resolution)
    grid_lat = np.linspace(lat_min, lat_max, resolution)
    grid_lon, grid_lat = np.meshgrid(grid_lon, grid_lat)

    grid_depth = griddata(
        (lon_bath, lat_bath),
        depth_bath,
        (grid_lon, grid_lat),
        method="linear",
    )
    grid_depth = np.where(grid_depth == -9999, np.nan, grid_depth)
    return grid_lon, grid_lat, grid_depth


def load_euc_campaign_points(euc_dir):
    """Load campaign TSV files and return seasonal profile-mean MiP points."""
    filenames = [
        "M117_EUC.tsv",
        "M181_EUC.tsv",
        "M158_EUC.tsv",
        "M147_EUC.tsv",
        "M174_EUC.tsv",
        "M159_EUC.tsv",
        "SO284_EUC.tsv",
        "AMAZOMIX_EUC.tsv",
    ]

    standard_columns = [
        "depth_bin",
        "Temp [°C]",
        "Sal",
        "Profile",
        "Date_Time",
        "Project",
        "profile",
        "Latitude",
        "Longitude",
        "Flux_mgC_m2",
        "MiP_abun",
        "MaP_abun",
        "MiP_biov",
        "MaP_biov",
        "diffPSD slope",
        "diffPSD slope stde",
        "dens_tmp",
    ]

    frames = []
    os.chdir(euc_dir)
    for filename in filenames:
        df = pd.read_csv(filename, sep="\t")
        frames.append(df[standard_columns])

    combined_df = pd.concat(frames, ignore_index=True)
    combined_df["Date_Time"] = pd.to_datetime(combined_df["Date_Time"])
    combined_df["Month"] = combined_df["Date_Time"].dt.month

    metadata = combined_df[["Profile", "Date_Time", "Month"]].drop_duplicates()
    numeric_df = combined_df.drop(columns=["Date_Time", "Project", "profile"])
    profile_means = numeric_df.groupby("Profile").mean(numeric_only=True).reset_index()
    merged_data = pd.merge(metadata, profile_means, on=["Profile", "Month"], how="inner")

    merged_data["Date_Time"] = pd.to_datetime(merged_data["Date_Time"])
    merged_data["Season"] = merged_data["Date_Time"].apply(get_real_season)

    season_fall = merged_data[merged_data["Season"] == "Fall"]
    season_spring = merged_data[merged_data["Season"] == "Spring"]
    return season_fall, season_spring


def load_rivers_gdf(rivers_gdb_dir):
    """Load and subset HydroRIVERS for map overlays."""
    os.chdir(rivers_gdb_dir)
    rivers_gdf = gpd.read_file("HydroRIVERS_v10_sa.gdb", engine="pyogrio")
    return rivers_gdf.query("ORD_FLOW <= 3")


def process_glorys_season(season_dir):
    """Compute density-filtered mean currents and speed from seasonal GLORYS files."""
    os.chdir(season_dir)
    nc_files = sorted([file for file in os.listdir() if file.endswith(".nc")])
    if not nc_files:
        raise FileNotFoundError(f"No .nc files found in {season_dir}")

    so_list = []
    thetao_list = []
    uo_list = []
    vo_list = []

    for filename in nc_files:
        print(f"Processing file: {filename}")
        dataset = xr.open_dataset(filename)

        so = dataset.variables["so"].values
        thetao = dataset.variables["thetao"].values
        uo = dataset.variables["uo"].values
        vo = dataset.variables["vo"].values

        lat = dataset.variables["latitude"].values
        lon = dataset.variables["longitude"].values
        depth = dataset.variables["depth"].values

        # Mean over time for each file
        so_list.append(np.nanmean(so, axis=0))
        thetao_list.append(np.nanmean(thetao, axis=0))
        uo_list.append(np.nanmean(uo, axis=0))
        vo_list.append(np.nanmean(vo, axis=0))

    so_data = np.stack(so_list, axis=0)
    thetao_data = np.stack(thetao_list, axis=0)
    uo_data = np.stack(uo_list, axis=0)
    vo_data = np.stack(vo_list, axis=0)

    depth_3d = depth[:, np.newaxis, np.newaxis]
    lon_3d = lon[np.newaxis, np.newaxis, :]
    lat_3d = lat[np.newaxis, :, np.newaxis]

    abs_psal = gsw.SA_from_SP(so_data, depth_3d, lon_3d, lat_3d)
    cons_temp = gsw.CT_from_t(abs_psal, thetao_data, depth_3d)
    dens = gsw.density.sigma0(abs_psal, cons_temp)

    dens_masked = np.where((dens < DENSITY_MIN) | (dens > DENSITY_MAX), np.nan, dens)
    uo_data[np.isnan(dens_masked)] = np.nan
    vo_data[np.isnan(dens_masked)] = np.nan

    u_mean = np.nanmean(uo_data, axis=0)
    v_mean = np.nanmean(vo_data, axis=0)

    # Depth-average to 2D map fields
    u2d = np.nanmean(u_mean, axis=0)
    v2d = np.nanmean(v_mean, axis=0)
    intensity = np.sqrt(u2d ** 2 + v2d ** 2)

    return lon, lat, u2d, v2d, intensity


def style_map_axis(ax, extent, rivers_gdf, show_left_labels=True):
    """Apply common map styling to one panel."""
    lon_min, lon_max, lat_min, lat_max = extent
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.add_geometries(rivers_gdf.geometry, crs=ccrs.PlateCarree(), edgecolor="#487bb6", linewidth=0.75)

    gl = ax.gridlines(draw_labels=True, linestyle="--", color="gray", alpha=0.5)
    gl.bottom_labels = False
    gl.right_labels = False
    gl.left_labels = show_left_labels
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {"size": 8, "color": "black"}
    gl.ylabel_style = {"size": 8, "color": "black"}


def plot_panel(
    ax,
    lon,
    lat,
    u,
    v,
    intensity,
    season_points,
    rivers_gdf,
    bathy_grid,
    title,
    show_left_labels,
    intensity_contour_level,
):
    """Render one seasonal panel: currents + MiP scatter + overlays."""
    lon2d, lat2d = np.meshgrid(lon, lat)

    im = ax.pcolormesh(lon2d, lat2d, intensity, cmap="Blues", vmin=0, vmax=1.6)

    scale_factor = 0.05
    u_scaled = u * scale_factor
    v_scaled = v * scale_factor
    skip = (slice(None, None, 10), slice(None, None, 10))
    ax.quiver(
        lon2d[skip],
        lat2d[skip],
        u_scaled[skip],
        v_scaled[skip],
        color="black",
        scale=2,
        width=0.003,
        headwidth=4,
    )

    ax.contour(
        lon2d,
        lat2d,
        intensity,
        colors="blue",
        levels=[intensity_contour_level],
        linewidths=0.4,
        alpha=0.7,
        transform=ccrs.PlateCarree(),
    )

    grid_lon, grid_lat, grid_depth = bathy_grid
    bath_contours = ax.contour(
        grid_lon,
        grid_lat,
        grid_depth,
        colors="blue",
        levels=np.arange(0, 500, 50),
        linewidths=0.4,
        alpha=0.7,
        transform=ccrs.PlateCarree(),
    )
    ax.clabel(
        bath_contours,
        inline=True,
        fontsize=8,
        fmt="%d",
        colors="black",
        inline_spacing=10,
    )

    sc = ax.scatter(
        season_points["Longitude"],
        season_points["Latitude"],
        c=season_points["MiP_abun"],
        transform=ccrs.PlateCarree(),
        s=30,
        cmap="viridis",
        vmin=0,
        vmax=70,
        edgecolor="k",
        linewidth=0.3,
    )

    style_map_axis(ax, MAP_EXTENT, rivers_gdf, show_left_labels=show_left_labels)
    ax.set_title(title)
    return im, sc


def build_figure():
    """Build and save Fig04 with Fall and Spring panels."""
    bathy_grid = load_bathymetry_grid(DATA_DIR / "bathy_Atlantic.csv", MAP_EXTENT, resolution=200)
    season_fall, season_spring = load_euc_campaign_points(EUC_DIR)
    rivers_gdf = load_rivers_gdf(RIVERS_GDB_DIR)

    lon_fall, lat_fall, u_fall, v_fall, intensity_fall = process_glorys_season(SAT_FALL_DIR)
    lon_spring, lat_spring, u_spring, v_spring, intensity_spring = process_glorys_season(SAT_SPRING_DIR)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(9, 12),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )

    im_fall, sc_fall = plot_panel(
        axes[0],
        lon_fall,
        lat_fall,
        u_fall,
        v_fall,
        intensity_fall,
        season_fall,
        rivers_gdf,
        bathy_grid,
        title="Fall",
        show_left_labels=True,
        intensity_contour_level=0.20,
    )

    im_spring, sc_spring = plot_panel(
        axes[1],
        lon_spring,
        lat_spring,
        u_spring,
        v_spring,
        intensity_spring,
        season_spring,
        rivers_gdf,
        bathy_grid,
        title="Spring",
        show_left_labels=False,
        intensity_contour_level=0.15,
    )

    fig.subplots_adjust(right=0.8)
    mip_cbar_ax = fig.add_axes([0.997, 0.4, 0.01, 0.18])
    mip_cbar = fig.colorbar(sc_spring, cax=mip_cbar_ax)
    mip_cbar.set_label("Mip \n (# L$^{-1}$)")

    vel_cbar_ax = fig.add_axes([0.2, 0.32, 0.7, 0.01])
    vel_cbar = fig.colorbar(im_spring, cax=vel_cbar_ax, orientation="horizontal")
    vel_cbar.set_label("Ocean current velocity  [m/s]")

    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_path = OUT_DIR / "Fig04.png"
    #plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure to: {fig_path}")
    #plt.close()


if __name__ == "__main__":
    build_figure()
