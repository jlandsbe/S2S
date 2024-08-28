import xarray as xr
import pandas as pd
import os as os
# Directory containing the NetCDF files
for var in ["PRECT","TEMP"]:
    file_name = f"ERA5_{var}_combined.nc"
    directory = '/scratch/jlandsbe/WeightedMaskAnalogForecasting-main/ERA5/' + var
    # List all NetCDF files in the directory
    # List all NetCDF files in the directory
    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.nc')]

    # Process each file: open, take the mean over time, assign the date, and store the result
    datasets = []
    dates = []

    for f in files:
        ds = xr.open_dataset(f)
        # Convert to cfdatetime
        date = ds.time.values[0].astype('datetime64[D]')
        ds_mean = ds.mean(dim='time')  # Take mean over time dimension
        ds_mean = ds_mean.assign_coords(date=(['date'], [date]))  # Assign the cfdatetime object as a coordinate
        datasets.append(ds_mean)
        dates.append(date)

    # Concatenate datasets along the new 'date' coordinate
    combined = xr.concat(datasets, dim='date')

    # Ensure 'date' coordinate is sorted
    sorted_dates = sorted(dates)
    combined = combined.reindex(date=sorted_dates)

    # Save the combined and sorted dataset to a new NetCDF file
    combined.to_netcdf(file_name)
