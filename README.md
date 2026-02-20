# TRATLEQ12_26

Code and data workflow used to generate manuscript figures for the TRATLEQ paper.

## Repository contents

- FIG01.py
- FIG02.py
- FIG03.py
- FIG04.py

These scripts generate publication figures and save them to the manuscript output directory.

## Figure scripts

- FIG01.py: Multi-panel figure combining satellite SST, satellite Chl-a, in situ CTD temperature sections, and in situ HPLC Chl-a sections.
- FIG02.py: M158 and M181 section plots for Mip, Map, POC flux, and POC flux anomaly.
- FIG03.py: Mip and zonal velocity section comparison (M158 vs M181), including EUC-core overlays.
- FIG04.py: Seasonal circulation maps (Fall and Spring) from GLORYS, overlaid with MiP observations and bathymetry/rivers.

## Data dependencies

The scripts use absolute paths and expect local data to exist in the TRATLEQ data tree.

Typical required inputs include:
- netCDF files (CTD/LADCP/satellite fields)
- CSV files (interpolated sections, trajectories, bathymetry)
- TSV files (EUC campaign merged datasets)
- XLSX files (HPLC logs)
- HydroRIVERS geodatabase


## Notes for reproducibility

- Scripts are configured with manuscript-specific absolute paths.
- Ensure all expected input files exist before running.
- If moving to another machine, update path constants and path strings in scripts.
- Some scripts load large 3D/4D arrays; sufficient memory is required.

