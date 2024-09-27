import xarray as xr
import numpy as np
import glob
import os

# # Paths to your netCDF files and output directory
# file_pattern = '/barnes-scratch/DATA/CESM2-LE/processed_data/monthly/TREFHT/*.nc'
# output_directory = '/barnes-scratch/DATA/CESM2-LE/processed_data/monthly/Detrended_TREFHT'


# # Get a list of all netCDF files
# files = glob.glob(file_pattern)

# # Open all datasets and concatenate into one
# all_datasets = xr.open_mfdataset(files, concat_dim='time', combine='nested', parallel=True, engine='h5netcdf')

# #could possibly add chunking: chunk_sizes = {'time': -1, 'lat': 1, 'lon': 1}

# # Extract the variable of interest
# temperature_all = all_datasets['TREFHT']


# # Group by month and calculate the mean for each month at each lat/lon coordinate
# daily_mean_all = temperature_all.groupby('time').mean(dim='time')

# # Process each file individually
# for file in files:
#     # Open the dataset
#     ds = xr.open_dataset(file)
    
#     # Extract the variable of interest
#     temperature = ds['TREFHT']
    
#     # Subtract the global monthly mean from the data
#     detrended_temperature = temperature.groupby('time') - daily_mean_all
#         # Create a new dataset with only the detrended TREFHT variable
#     detrended_ds = xr.Dataset({'TREFHT': detrended_temperature})
    
#     # Save the detrended data to a new netCDF file
#     output_file = os.path.join(output_directory, os.path.basename(file))
#     detrended_ds.to_netcdf(output_file)
#     detrended_ds.close()
#     # Close the dataset to free up resources
#     ds.close()

# print("Detrending Tempreature completed and files saved.")


# # Paths to your netCDF files and output directory
# file_pattern = '/barnes-scratch/DATA/CESM2-LE/processed_data/monthly/PRECT/*.nc'
# output_directory = '/barnes-scratch/DATA/CESM2-LE/processed_data/monthly/Detrended_PRECT'


# # Get a list of all netCDF files
# files = glob.glob(file_pattern)

# # Open all datasets and concatenate into one
# all_datasets = xr.open_mfdataset(files, concat_dim='time', combine='nested', parallel=True, engine='h5netcdf')

# #could possibly add chunking: chunk_sizes = {'time': -1, 'lat': 1, 'lon': 1}

# # Extract the variable of interest
# temperature_all = all_datasets['PRECT']


# # Group by month and calculate the mean for each month at each lat/lon coordinate
# daily_mean_all = temperature_all.groupby('time').mean(dim='time')

# # Process each file individually
# for file in files:
#     # Open the dataset
#     ds = xr.open_dataset(file)
    
#     # Extract the variable of interest
#     temperature = ds['PRECT']
    
#     # Subtract the global monthly mean from the data
#     detrended_temperature = temperature.groupby('time') - daily_mean_all
#         # Create a new dataset with only the detrended TREFHT variable
#     detrended_ds = xr.Dataset({'PRECT': detrended_temperature})
    
#     # Save the detrended data to a new netCDF file
#     output_file = os.path.join(output_directory, os.path.basename(file))
#     detrended_ds.to_netcdf(output_file)
#     detrended_ds.close()
#     # Close the dataset to free up resources
#     ds.close()

# print("Detrending Precipitation completed and files saved.")


# Paths to your netCDF files and output directory
file_pattern = '/barnes-scratch/DATA/CESM2-LE/processed_data/monthly/U/*.nc'
output_directory = '/barnes-scratch/DATA/CESM2-LE/processed_data/monthly/Detrended_U'


# Get a list of all netCDF files
files = glob.glob(file_pattern)

# Open all datasets and concatenate into one
all_datasets = xr.open_mfdataset(files, concat_dim='time', combine='nested', parallel=True, engine='h5netcdf')

#could possibly add chunking: chunk_sizes = {'time': -1, 'lat': 1, 'lon': 1}

# Extract the variable of interest
temperature_all = all_datasets['U']


# Group by month and calculate the mean for each month at each lat/lon coordinate
daily_mean_all = temperature_all.groupby('time').mean(dim='time')

# Process each file individually
for file in files:
    # Open the dataset
    ds = xr.open_dataset(file)
    
    # Extract the variable of interest
    temperature = ds['U']
    
    # Subtract the global monthly mean from the data
    detrended_temperature = temperature.groupby('time') - daily_mean_all
        # Create a new dataset with only the detrended TREFHT variable
    detrended_ds = xr.Dataset({'U': detrended_temperature})
    
    # Save the detrended data to a new netCDF file
    output_file = os.path.join(output_directory, os.path.basename(file))
    detrended_ds.to_netcdf(output_file)
    detrended_ds.close()
    # Close the dataset to free up resources
    ds.close()

print("Detrending U250 completed and files saved.")

