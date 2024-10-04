"""Data processing.

Functions
---------

"""

import numpy as np
import regions
import xarray as xr
import metrics
import base_directories
import pickle
import gzip
import os
import warnings
from datetime import timedelta
import regionmask
import gc

import os
import grp
import os
import grp

# Get your user ID
user_id = os.getuid()

# Get your username from the environment
user_name = os.environ.get('USER') or os.environ.get('LOGNAME')

# Fetch the groups you're part of
groups = [g.gr_name for g in grp.getgrall() if user_name in g.gr_mem]

# Also include your primary group
primary_group = grp.getgrgid(os.getgid()).gr_name
groups.append(primary_group)

print(f"User {user_name} belongs to the following groups: {groups}")

print(f"User {user_name} belongs to the following groups: {groups}")

__author__ = "Jamin K. Rader, Elizabeth A. Barnes, Randal J. Barnes, and Jacob Landsberg"
__version__ = "June 2023"

dir_settings = base_directories.get_directories()


def build_data(settings, data_directory):
    print(settings["analog_members"])
    #grab the experiment and check if we already have created the data for it, if so use that data, and don't recalculate persistance (just set as 1)
    if "base_exp_name" in settings.keys():
        data_exp_name = settings["base_exp_name"]
    else:
        data_exp_name = settings["exp_name"]

    if settings["presaved_data_filename"] is None:
        data_savename = dir_settings["data_directory"] + 'presaved_data_' + data_exp_name + '.pickle'
        persist_savename = dir_settings["data_directory"] + data_exp_name + '_persist_error' + '.pickle'
        analog_dates_savename = dir_settings["data_directory"] + data_exp_name + '_analog_dates' + '.pickle'
        soi_dates_savename = dir_settings["data_directory"] + data_exp_name + '_soi_dates' + '.pickle'
    else:
        data_savename = dir_settings["data_directory"]+settings["presaved_data_filename"]
        persist_savename = dir_settings["data_directory"]+settings["presaved_data_filename"] + '_persist_error.pickle'
        analog_dates_savename = dir_settings["data_directory"]+settings["presaved_data_filename"] + '_analog_dates.pickle'
        soi_dates_savename = dir_settings["data_directory"]+settings["presaved_data_filename"] + '_soi_dates.pickle'
    persist_err=1 #setting to 1, could save and load this actual value if necessary

#if 
    if os.path.exists(data_savename) is False:
        print('building the data from netcdf files')

        # initialize empty dictionaries to old the standardization info from the training data
        input_standard_dict = {
            "ens_mean": None,
            "data_mean": None,
            "data_std": None,
        }
        output_standard_dict = {
            "ens_mean": None,
            "data_mean": None,
            "data_std": None,
        }
        #analogs, and SOIs for training, validation and testing are all preprocessed identically
        #get the data for our analog library
        print('getting analog pool...')
        analog_input, analog_output, input_standard_dict, output_standard_dict, __, tethers_analogs, progressions_analogs = process_input_output(
            data_directory, settings, members=settings["analog_members"], input_standard_dict=input_standard_dict,
            output_standard_dict=output_standard_dict,
        )
        if settings["split_soi"]:
            year_0 = settings["years"]
            midpoint = year_0[0]+((year_0[1]-year_0[0])/2)
            settings["years"] = (year_0[0], midpoint)
        if settings["date_info"]:
            analog_dates = np.tile(analog_input['time'].values,len(settings["analog_members"]))#saving dates of analog inputs
        else: 
            analog_dates = None

        print('getting soi training data...')
        #get the data for the SOIs to train on 
        soi_train_input, soi_train_output, __, __, __, tethers_soi,__ = process_input_output(
            data_directory, settings, members=settings["soi_train_members"], input_standard_dict=input_standard_dict,
            output_standard_dict=output_standard_dict,
        )
        if settings["split_soi"]:
            settings["years"] = year_0
         #get the data for the SOIs to validate on 
        print('getting validation data...')
        soi_val_input, soi_val_output, __, __, __, __,__ = process_input_output(
            data_directory, settings, members=settings["soi_val_members"], input_standard_dict=input_standard_dict,
            output_standard_dict=output_standard_dict,
        )
        #get the data for the SOIs to test on 
        if settings["split_soi"]:
            settings["years"] = (midpoint, year_0[1])
        print('getting testing data...')
        soi_test_input, soi_test_output, __, __, persist_err, __, progressions_soi = process_input_output(
            data_directory, settings, members=settings["soi_test_members"], input_standard_dict=input_standard_dict,
            output_standard_dict=output_standard_dict,
        )
        if settings["date_info"]:
            soi_dates = np.tile(soi_test_input['time'].values,len(settings["analog_members"])) #saving dates of analog inputs
        else:
            soi_dates = None
        #here we are stacking the data to reduce the dimensions from time and member number to just sample number (we aren't concerned with differentiating those)
        #the output will either be a single number per sample if targ_scalar is specified to predict the average over the region, otherwise it will keep lat x lon
        for idx, t in enumerate(tethers_analogs):
            tethers_analogs[idx] = stack_to_samples([], t, settings["targ_scalar"], no_inputs = 1)
        for idx, t in enumerate(tethers_soi):
            tethers_soi[idx] = stack_to_samples([], t, settings["targ_scalar"], no_inputs = 1)
        for idx, p in enumerate(progressions_analogs):
            progressions_analogs[idx] = stack_to_samples(p,[], settings["targ_scalar"], no_outputs= 1)
        for idx, p in enumerate(progressions_soi):
            progressions_soi[idx] = stack_to_samples(p,[], settings["targ_scalar"], no_outputs = 1)
        analog_input, analog_output = stack_to_samples(analog_input, analog_output, settings["targ_scalar"])
        soi_train_input, soi_train_output = stack_to_samples(soi_train_input, soi_train_output, settings["targ_scalar"])
        soi_val_input, soi_val_output = stack_to_samples(soi_val_input, soi_val_output, settings["targ_scalar"])
        soi_test_input, soi_test_output = stack_to_samples(soi_test_input, soi_test_output, settings["targ_scalar"])

        lat = analog_input["lat"] #keep
        lon = analog_input["lon"] #keep

        area_weights = np.abs(np.cos(np.deg2rad(lat))).to_numpy() #keep
###### everything below here seems fine, just loading and saving data, etc.
        print(f"saving the pre-saved training/validation/testing data.")
    #save and then reload the data
        print(f"   {data_savename}")
        with gzip.open(data_savename, "wb") as fp:
            pickle.dump(analog_input.to_numpy().astype(np.float32), fp)
            pickle.dump(analog_output.to_numpy().astype(np.float32), fp)

            pickle.dump(soi_train_input.to_numpy().astype(np.float32), fp)
            pickle.dump(soi_train_output.to_numpy().astype(np.float32), fp)

            pickle.dump(soi_val_input.to_numpy().astype(np.float32), fp)
            pickle.dump(soi_val_output.to_numpy().astype(np.float32), fp)

            pickle.dump(soi_test_input.to_numpy().astype(np.float32), fp)
            pickle.dump(soi_test_output.to_numpy().astype(np.float32), fp)

            pickle.dump(input_standard_dict, fp)
            pickle.dump(output_standard_dict, fp)

            pickle.dump(lat.to_numpy().astype(np.float32), fp)
            pickle.dump(lon.to_numpy().astype(np.float32), fp)

            pickle.dump(area_weights.astype(np.float32), fp)
            for t in tethers_analogs:
                pickle.dump(np.squeeze(t.to_numpy().astype(np.float32)), fp)
            for t in tethers_soi:
                pickle.dump(np.squeeze(t.to_numpy().astype(np.float32)), fp)
            for p in progressions_analogs:
                pickle.dump(np.squeeze(p.to_numpy().astype(np.float32)), fp)
            for p in progressions_soi:
                pickle.dump(np.squeeze(p.to_numpy().astype(np.float32)), fp)
        with open(persist_savename, 'wb') as f:
            pickle.dump(persist_err, f)
        if settings["date_info"]:
            with open(analog_dates_savename, 'wb') as f:
                pickle.dump(analog_dates, f)
            with open(soi_dates_savename, 'wb') as f:
                pickle.dump(soi_dates, f)

    print(f"loading the pre-saved training/validation/testing data.")
    print(f"   {data_savename}")
    tethers_analogs = []
    tethers_soi = []
    progressions_analogs = []
    progressions_soi = []
    with gzip.open(data_savename, "rb") as fp:
        analog_input = pickle.load(fp)
        analog_output = pickle.load(fp)

        soi_train_input = pickle.load(fp)
        soi_train_output = pickle.load(fp)

        soi_val_input = pickle.load(fp)
        soi_val_output = pickle.load(fp)

        soi_test_input = pickle.load(fp)
        soi_test_output = pickle.load(fp)

        input_standard_dict = pickle.load(fp)
        output_standard_dict = pickle.load(fp)

        lat = pickle.load(fp)
        lon = pickle.load(fp)

        area_weights = pickle.load(fp)
        for idx in range(len(settings["tethers"])):
            t = pickle.load(fp)
            tethers_analogs.append(t) 
        for idx in range(len(settings["tethers"])):
            t = pickle.load(fp)
            tethers_soi.append(t) 
        for idx in range(len(settings["progressions"])):
            p = pickle.load(fp)
            progressions_analogs.append(p) 
        for idx in range(len(settings["progressions"])):
            p = pickle.load(fp)
            progressions_soi.append(p) 
    with open(persist_savename, 'rb') as f:
        persist_err = pickle.load(f)
    if not settings["date_info"]:
        analog_dates = None
        soi_dates = None   
    else: 
        with open(analog_dates_savename, 'rb') as f:
            analog_dates = pickle.load(f)
        with open(soi_dates_savename, 'rb') as f:
            soi_dates = pickle.load(f)

    # summarize the data
    analog_text = ("   analog data\n"
                   f" # analog samples = {analog_output.shape[0]}\n"
                   )
    train_text = ("   training data\n"
                  f" # soi samples = {soi_train_output.shape[0]}\n"
                  )
    val_text = ("   validation data\n"
                f"   # soi samples = {soi_val_output.shape[0]}\n"
                )
    test_text = ("   testing data\n"
                 f"  # soi samples = {soi_test_output.shape[0]}\n"
                 )
    print(analog_text + train_text + val_text + test_text)
    return (analog_input, analog_output, soi_train_input, soi_train_output, soi_val_input, soi_val_output,
            soi_test_input, soi_test_output, input_standard_dict, output_standard_dict, lat, lon, persist_err, analog_dates, soi_dates, tethers_analogs, tethers_soi, progressions_analogs, progressions_soi)



def process_input_output(data_directory, settings, input_standard_dict=None, 
                         output_standard_dict=None, obs_info=None, members=None):

    # get the input (feautre) and output (target) data as xarrays
    data_target = get_netcdf(settings["target_var"], data_directory, settings, members=members)
    data_feature = get_netcdf(settings["feature_var"], data_directory, settings, members=members)
    #if you are using 2 feature to predict, get the extra channel
    if settings["extra_channel"] != None:
        extra_channel = []
        for chan_idx, chan_nam in enumerate(settings["extra_channel"]):
            extra_channel.append(get_netcdf(chan_nam, data_directory, settings, members=members, time_tendency=settings["time_tendency"][chan_idx]))
    else:
        extra_channel = None
    #optionally set your data to random values for testing purposes
    if settings["randomize"]:
        random_array_tar = 5000*(np.random.rand(data_target.sizes['time'],data_target.sizes['lat'], data_target.sizes['lon'])-.5)
        random_array_feat = 5000*(np.random.rand(data_feature.sizes['time'],data_feature.sizes['lat'], data_feature.sizes['lon'])-.5)
        data_target = data_target + random_array_tar
        data_feature = data_feature + random_array_feat

    #if you are using a ocean/land only variable, then you can mask out the respective regions values (sets to 0)
    if settings["maskout_landocean_output"] is not None:
        data_target = mask_in_land_ocean(data_target, settings, settings["maskout_landocean_output"]) 
    if settings["maskout_landocean_input"] is not None:
        data_feature = mask_in_land_ocean(data_feature, settings, settings["maskout_landocean_input"]) 
    data_feature, __, __ = extract_region(data_feature, regions.get_region_dict(settings["feature_region_name"]))
    data_output, __, __ = extract_region(data_target, regions.get_region_dict(settings["target_region_name"]))
    #optionally apply area weighting based on latitude and optionally average over target region, so you are predicting a single number for the whole region (detmerined by targ_scalar and error_calc).
    data_output = compute_global_mean(data_output, settings["targ_scalar"], settings["error_calc"])


    (data_input, data_output, input_standard_dict, output_standard_dict, tethers, progressions) = process_data(data_feature, data_output, settings, input_standard_dict, output_standard_dict, extra_channel)
    persist_err = np.mean(compute_persistance(data_output, settings, settings["lead_time"], settings["error_calc"]))

    gc.collect()
    return data_input, data_output, input_standard_dict, output_standard_dict, persist_err, tethers, progressions

def compute_persistance(output_array, settings, leadtime, error_type = "mse"):
    tot_lead = leadtime + int((settings["smooth_len_input"]/settings["data_gap"]))
    array_past = output_array.shift(time = tot_lead).dropna(dim="time")
    array_fut = output_array[:, tot_lead:]
    return metrics.get_persist_errors(array_fut.to_numpy(), array_past.to_numpy(), error_type)

#we change our data from having a member and time dimension, to just a sample dimension, since we aren't concerned with differentiating by member
def stack_to_samples(data_input, data_output, scalar_mode, no_inputs = 0, no_outputs = 0):
    if no_outputs:
        return data_input.stack(sample=("member", "time")).transpose("sample", "lat", "lon", "channel", )
    if no_inputs:
        if scalar_mode:
            return data_output.stack(sample=("member", "time")).transpose("sample")
        else:
            print("")
            return data_output.stack(sample=("member", "time")).transpose("sample", "lat", "lon")
    if scalar_mode:
        return (data_input.stack(sample=("member", "time")).transpose("sample", "lat", "lon", "channel", ), 
            data_output.stack(sample=("member", "time")).transpose( "sample"))
    else:
        return (data_input.stack(sample=("member", "time")).transpose("sample", "lat", "lon", "channel", ), 
            data_output.stack(sample=("member", "time")).transpose( "sample", "lat", "lon"))


def repeat_soi_members(soi_input, soi_output, n_members_analog):
    # check that the soi have the same number of members as the analog by copying
    if soi_input["member"].shape[0] != n_members_analog:
        n_members_soi = soi_input["member"].shape[0]
        print(f"  {n_members_soi} members in soi != {n_members_analog} members in analog, repeating soi members...")
        print(f"  before soi_data.shape = {soi_input.shape}")

        assert n_members_analog % n_members_soi == 0, "number of members in the analog set is not divisible by the " \
                                                      "number of members in the soi set"

        n_repeats = n_members_analog / n_members_soi
        soi_input_repeat = soi_input
        soi_output_repeat = soi_output
        for icycle in np.arange(0, n_repeats - 1):
            soi_input_repeat = xr.concat((soi_input_repeat, soi_input), dim="member")
            soi_output_repeat = xr.concat((soi_output_repeat, soi_output), dim="member")

        print(f"  after soi_data.shape = {soi_input_repeat.shape}")
        return soi_input_repeat, soi_output_repeat
    else:
        return soi_input, soi_output

# smooth the data, align the data, and add a secondary feature channel if applicable
def process_data(data_feature, data_target, settings, input_standard_dict, output_standard_dict, extra_chan = None):
    #here we add a channel dimension and add (if applicable) an extra feature channel
    data_input = add_extra_channel(data_feature, settings, extra_chan) 
    #if averaged_days is 0, then perform a running mean to smooth the input/output data
    if settings["averaged_days"] == 0:
        if settings["smooth_len_input"] != 0:
            data_input = smooth_data(data_input, settings["smooth_len_input"])
        if settings["smooth_len_output"] != 0:
            data_output = smooth_data(data_target, settings["smooth_len_output"])
        else:
            data_output = data_target
    #otherwise, smooth via downscaling (preserve more independence between samples) - this must be a multiple of the lead time (e.g. you cannot have weekly samples with a 8 day lead time
        # but you could with a 7, 14, or 21 day lead. )
    else:
        #this is more of a check to make sure we have the same dates for input/output, should be ok to remove
        data_input, data_target = xr.align(data_input, data_target, join="inner", exclude = ("lat","lon"))
        t_avg = str(settings["averaged_days"]) + "M"
        #here we shift forward the data and then remove the 1st time step to effectively compute a backward downsample (e.g. Jan 1 - Jan 7 data will all go to Jan 1, not all to Jan 7)
        data_input = data_input.resample(time = t_avg).mean().shift(time=1)
        data_input = data_input[:,1:]
        data_output = data_target.resample(time = t_avg).mean()
        data_output.dropna("time")
        if settings["lead_time"]%settings["averaged_days"]:
            raise ValueError("lead_time must be divisible by averaged_days")

    #align our data, so all dates are lined up again before applying the lead time filter. This results in index i having data_input from lead time earlier than the data_output
    #this is used to filter out any years that don't have correspinding years in the input/output
    data_input_orig, data_output_orig = xr.align(data_input, data_output, join="inner", exclude = ("lat","lon")) 
    data_input_orig = gap_data(data_input_orig, settings["data_gap"]) 
    data_output_orig = gap_data(data_output_orig, settings["data_gap"])
    #get any tethers you want:
    tethers = []
    for idx, lead_amount in enumerate(settings["tethers"]):
        __, tether_i= filter_lead_times(data_input_orig, data_output_orig, lead_amount)
        tethers.append(tether_i)

     #after this point, don't realign based on time or you'll get rid of lead time
    data_input, data_output = filter_lead_times(data_input_orig, data_output_orig, settings["lead_time"])
    for idx,t in enumerate(tethers):
        t = t[0:len(data_input.time)]
        tethers[idx] = t
    
    #filter out any years/months that you don't care about seeing in the data 
    #(based on the targets, so if you filtered january out, you would have no target January, but you would use a feautre January to predict a target February)
    if settings["years"] != None:
        data_input, data_output, tethers = filter_years(data_input, data_output, settings["years"], tethers)
    if settings["months"] != None:
        data_input, data_output,tethers = filter_months(data_input, data_output, settings["months"], tethers)
    #progressions, data_input, data_output = filter_progression_leads(data_input_orig, data_input, data_output, settings["progressions"])
    #Here we optionally standardize the data over all samples (i.e. standardize each grid point)
    progressions = []
    data_input, input_standard_dict = standardize_data(data_input, input_standard_dict, settings["standardize_bool"])
    data_output, output_standard_dict = standardize_data(data_output, output_standard_dict,settings["standardize_bool"], settings["median"])

    for idx, t in enumerate(tethers):
        tethers[idx] = standardize_data(t, None, settings["standardize_bool"], settings["median"])[0]

    for idx, p in enumerate(progressions):
        progressions[idx] = standardize_data(p, None, settings["standardize_bool"], settings["median"])[0]

    return data_input, data_output, input_standard_dict, output_standard_dict, tethers, progressions

#cut out any years you don't want
def filter_years(data_input, data_output, years, tethers =[]):
    years = np.arange(years[0],years[1],1)
    itime = np.where(np.isin(data_input.time.dt.year.values, years))[0]
    d_inp = data_input[:, itime, :, :, :]
    d_out = data_output[:, itime]
    for idx, t in enumerate(tethers):
        tethers[idx] = t[:,itime]
    return d_inp, d_out, tethers

#cut out any months you don't want
def filter_months(data_input, data_output, months, tethers = []):
    itime = np.where(np.isin(data_input.time.dt.month.values, months))[0]
    d_inp = data_input[:, itime, :, :, :]
    d_out = data_output[:, itime]
    for idx, t in enumerate(tethers):
        tethers[idx] = t[:,itime]
    return d_inp, d_out, tethers
    
#add in shift of lead times between input and output
def filter_lead_times(data_input, data_output, lead_time):
    # Drop the last lead_time data points in the time dimension for data_input
    data_input_filtered = data_input.isel(time=slice(None, -lead_time))
    
    # Drop the first lead_time data points in the time dimension for data_output
    data_output_filtered = data_output.isel(time=slice(lead_time, None))
    
    # Return filtered data_input and data_output
    return data_input_filtered, data_output_filtered

def filter_progression_leads(data_in_orig, data_input_filtered, data_output, progression_leads = []):
    progressions = []
    lengths = np.zeros(len(progression_leads))
    for idx, t_lead in enumerate(progression_leads):
        times_prog = data_input_filtered.time + timedelta(days=-t_lead)
        prog = data_in_orig.where(data_in_orig.time.isin(times_prog.values), drop=True)
        lengths[idx] = len(prog.time)
        progressions.append(prog)
    if len(progression_leads)>0:
        true_len = int(np.min(lengths))
    else:
        true_len=0
    for idx, p in enumerate(progressions):
        progressions[idx] = p[:,-true_len:,:,:]
     #Return filtered data_input and data_output
    return progressions, data_input_filtered[:,-true_len:], data_output[:,-true_len:]

def add_extra_channel(data_in, settings, extra_chan):
    if settings["extra_channel"] == None or settings["extra_channel"] == 0:
        return data_in.expand_dims(dim={"channel": 1}, axis=-1).copy()
    else:
        data_in = data_in.expand_dims(dim={"channel": 1}, axis=-1)
        for idx,__ in enumerate(settings["extra_channel"]):
            data_in = xr.concat([data_in, extra_chan[idx]], dim = "channel")
        # d_present, d_past = xr.align(data_in[:, settings["extra_channel"]:, :, :],
        #                              data_in[:, :-settings["extra_channel"], :, :], join="override", copy=True)
        # data_in = d_present.expand_dims(dim={"channel": 2}, axis=-1).copy()
        # data_in[:, :, :, :, 1] = data_in[:, :, :, :, 0] - d_past

        return data_in

#this keeps times the same, but I guess then if you use indices in the future it will work ok?????
def smooth_data(data, smooth_time): 
    if smooth_time<0:
        data = data.rolling(time=-smooth_time).mean(skipna=True)
        data = data.shift(time = smooth_time + 1)
        data = data.dropna("time")
    elif smooth_time>0:
        data = data.rolling(time=smooth_time).mean(skipna=True)
        data = data.dropna("time")
    
    return data

def gap_data(data, step): 
    if step < 1:
        raise ValueError("Step must be a positive integer")
    # Select every x-th entry in the dataset
    data = data.isel(time=slice(None, None, step))
    return data




def extract_region(data, region=None, lat=None, lon=None, mask_builder = 0, progressions = 0):
    if region is None:
        min_lon, max_lon = [0, 360]
        min_lat, max_lat = [-90, 90]
    else:
        min_lon, max_lon = region["lon_range"]
        min_lat, max_lat = region["lat_range"]
    if mask_builder:
        ilon = np.where((lon >= min_lon) & (lon <= max_lon))[0]
        ilat = np.where((lat >= min_lat) & (lat <= max_lat))[0]
        data_regional = data.copy()
        weighting = np.sqrt(np.abs(np.cos(np.deg2rad(ilat))))
        data_regional[min(ilat):max(ilat)+1,min(ilon):max(ilon)+1,:] = 1
        data_regional[min(ilat):max(ilat)+1,min(ilon):max(ilon)+1,:] = data_regional[min(ilat):max(ilat)+1,min(ilon):max(ilon)+1,:] * weighting[:,np.newaxis,np.newaxis]
        return data_regional
    if isinstance(data, xr.DataArray): #if the data passed in is an xarray, filter out data not within the lon, lat bounds
        mask_lon = (data.lon >= min_lon) & (data.lon <= max_lon)
        mask_lat = (data.lat >= min_lat) & (data.lat <= max_lat)
        return data.where(mask_lon & mask_lat, drop=True), None, None
    if progressions:
        assert(len(data.shape) == 5), "expected np.array of len(shape)==5"
        ilon = np.where((lon >= min_lon) & (lon <= max_lon))[0]
        ilat = np.where((lat >= min_lat) & (lat <= max_lat))[0]
        data_masked = data[:, :, ilat, :, :]
        data_masked = data_masked[:, :, :, ilon, :]
        return data_masked
    else:
        assert len(data.shape) == 4, "expected np.array of len(shape)==4"
        ilon = np.where((lon >= min_lon) & (lon <= max_lon))[0]
        ilat = np.where((lat >= min_lat) & (lat <= max_lat))[0]
        data_masked = data[:, ilat, :, :]
        data_masked = data_masked[:, :, ilon, :]
        return data_masked, lat[ilat], lon[ilon]

def get_netcdf(var, data_directory, settings, members = [], obs_fn = None, time_tendency = 0):

    da_all = None

    for ens in members:
        member_text = str(ens)
        if member_text[0] == "s":
            member_text =  "smbb.LE2-"+ member_text[1:]
        if member_text[0] == "c":
            member_text =  "CMIP6.LE2-"+ member_text[1:]
        print('   ensemble member = ' + member_text)
        #fp = dir_settings["net_data"] + "/" + "Monthly_Temp/" + member_text + "." + var + ".1850-2100.shifted.nc" 
        fp = dir_settings["net_data"] + "/" + "Detrended_" + var +"/" + member_text + "." + var + ".1850-2100.shifted.nc"
        print(os.listdir("/barnes-scratch/")) 
        print(os.path.exists("/barnes-scratch/DATA/"))
        print(os.path.exists(fp))

        if ens == "ERA5":
            da = xr.open_dataset(fp)['__xarray_dataarray_variable__'].squeeze()
            da = da.convert_calendar("standard", use_cftime=True)
        else:
            da = xr.open_dataset(fp)[var].squeeze()
        da = da.fillna(0.0)
        if ens == "ERA5":
            da = da.convert_calendar("standard", use_cftime=True)
        if time_tendency:
            da = da.diff(dim = 'time', n = 1, label='lower')
        #if you don't already have members, make a member dimension, then after that add on the new data for each member
        if da_all is None:
            da_all = da.expand_dims(dim={"member": 1}, axis=0)
        else:
            da_all = xr.concat([da_all, da], dim="member")
        gc.collect()
    return da_all
#given a variable, get the xarray data and add a member dimension, that we will use instead of time/member number later

#probably have to change the directories here but then it's just multiplication
def mask_in_land_ocean(da, settings, maskin="land"):
    # if no land mask or ocean masks exists, run make_land_ocean_mask()
    if not os.path.isfile(dir_settings["net_data"] + "/month_ocean_mask.nc") or not os.path.isfile(dir_settings["net_data"]  + "/month_land_mask.nc") or not os.path.isfile(dir_settings["net_data"]  + "/month_no_mask.nc"):
        make_land_mask(settings)
    if maskin == "land":
        with gzip.open(dir_settings["net_data"] + "/month_land_mask.pickle", "rb") as fp:
            mask = pickle.load(fp)
    elif maskin == "ocean":
        with gzip.open(dir_settings["net_data"] + "/month_ocean_mask.pickle", "rb") as fp:
            mask = pickle.load(fp)
    elif maskin == "all":
            with gzip.open(dir_settings["net_data"] + "/month_no_mask.pickle", "rb") as fp:
                mask = pickle.load(fp)
    else:
        raise NotImplementedError("no such mask type.")
    if da is not None:
        return da*mask
    else:
        return mask


def make_land_mask(settings):
    x_data = xr.load_dataset("/barnes-scratch/DATA/CESM2-LE/raw_data/monthly/tos/b.e21.BHISTcmip6.f09_g17.LE2-1001.001.cam.h0.SST.185001-185912.nc")["SST"]
    x_data = x_data.mean(dim="time", skipna=True)
    
    #x_ocean keeps ocean values, sets land to 0
    x_ocean = xr.where(x_data.isnull(), 0.0, 1.0)#replace where there aren't values with 0 (since it's a map of SSS, this implies those areas are land)
    x_ocean.to_netcdf(dir_settings["net_data"] + "month_ocean_mask.nc")
    x_ocean.plot()

    #x_land keeps land values, sets ocean to 1
    x_land = xr.where(x_data.isnull(), 1.0, 0.0)#replace where there aren't values with 1 (since it's a map of SSS, this implies it is land)
    x_land.to_netcdf(dir_settings["net_data"] + "month_land_mask.nc")
    x_land.plot()

    #x_all keeps land and ocean values
    x_all = x_land + x_ocean
    x_all.to_netcdf(dir_settings["net_data"] + "month_no_mask.nc")
    x_all.plot()

    da = xr.load_dataarray(dir_settings["net_data"] + "month_land_mask.nc")
    data_savename = dir_settings["net_data"] + "month_land_mask.pickle"
    with gzip.open(data_savename, "wb") as fp:
        pickle.dump(da.to_numpy().astype(np.float32), fp)

    da = xr.load_dataarray(dir_settings["net_data"] + "month_no_mask.nc")
    data_savename = dir_settings["net_data"] + "month_no_mask.pickle"
    with gzip.open(data_savename, "wb") as fp:
        pickle.dump(da.to_numpy().astype(np.float32), fp)

    da = xr.load_dataarray(dir_settings["net_data"] + "month_ocean_mask.nc")
    data_savename = dir_settings["net_data"] + "month_ocean_mask.pickle"
    with gzip.open(data_savename, "wb") as fp:
        pickle.dump(da.to_numpy().astype(np.float32), fp)

    return 1


def standardize_data(data, standard_dict, standardize_bool, median=0):
    if type(standard_dict) == type(None):
        standard_dict = {
            "ens_mean": None,
            "data_mean": None,
            "data_std": None,
        }
    if standard_dict["data_mean"] is None:
        if standardize_bool:
            standard_dict["data_mean"] = data.mean(axis=(0, 1)).to_numpy().astype(np.float32) #this is the mean over the ensemble for all times
        else:
            standard_dict["data_mean"] = 0.

    if standard_dict["data_std"] is None:
        if standardize_bool:
            standard_dict["data_std"] = data.std(axis=(0, 1)).to_numpy().astype(np.float32) #this is the sd over the ensemble for all times per latitude
        else:
            standard_dict["data_std"] = 1.0
    if median: 
        standardized_data = ((data - data.median(axis=(0, 1)).to_numpy().astype(np.float32))/standard_dict["data_std"]).fillna(0.)
        return standardized_data, standard_dict
    standardized_data = ((data - standard_dict["data_mean"]) / standard_dict["data_std"]).fillna(0.)
    return standardized_data, standard_dict


#picks some random indices (batch size #) and then returns the soi input of those indices, the analog inputs, and the difference between those states
def batch_generator(settings, soi_input, soi_output, analog_input, analog_output, batch_size, rng_seed=33):

    rng = np.random.default_rng(rng_seed)

    while True:
        i_soi = rng.choice(np.arange(0, soi_input.shape[0]), batch_size, replace=True)
        i_analog = rng.choice(np.arange(0, analog_input.shape[0]), batch_size, replace=True)
        targets = metrics.get_targets(settings, soi_output[i_soi], analog_output[i_analog])
        if settings["weighted_train"] != 1:
            weights = 1/(targets+.1)+1
        else:
            weights = np.ones(np.shape(targets))
        yield [soi_input[i_soi, :, :, :], analog_input[i_analog, :, :, :]], [targets], weights

# # Building observational data
# def build_obs_data(settings, data_directory, obs_info):
#     # initialize empty dictionaries to old the standardization info from the training data
#     input_standard_dict = obs_info['input_standard_dict']
#     output_standard_dict = obs_info['output_standard_dict']
#     print('getting observations...')
#     obs_input, obs_output, input_standard_dict, output_standard_dict = process_input_output(
#         data_directory, settings, input_standard_dict, output_standard_dict, obs_info=obs_info
#     )

#     obs_input, obs_output = stack_to_samples(obs_input, obs_output)

#     lat = obs_input["lat"].to_numpy().astype(np.float32) 
#     lon = obs_input["lon"].to_numpy().astype(np.float32) 

#     area_weights = np.cos(np.deg2rad(lat))

#     return obs_input.to_numpy().astype(np.float32), \
#         obs_output.to_numpy().astype(np.float32), \
#         input_standard_dict, output_standard_dict, lat, lon
    

def compute_global_mean(data, single_target, target_type, lat=None):
    #if you are going to predict a map itself, then you don't necessarily need area weighting, since you'll return a map, so just return the data itself
    if target_type == "map":
        return data
    if isinstance(data, xr.DataArray):
        #set up cosine weighting based on latitude
        weights = np.cos(np.deg2rad(data.lat))
        weights.name = "weights"
        temp_weighted = data.weighted(weights)
        #if you are going to average over target region and return a single number, take mean over lat/lon
        if single_target:
            global_mean = temp_weighted.mean(("lon", "lat"), skipna=True) #computes a global mean of the data array (global meaning within the region defined earlier)

        #if you are going to be computing the ACC, then keep lat, lon but weight by cos(lat)^.5, since you will square the values
        elif target_type == "field":
            w = (np.broadcast_to(np.expand_dims(weights,axis=1), np.shape(data)[2:4]))
            return data * w**.5
        #otherwise (i.e. using mse but wanting to keep all values), then keep lat, lon and weight by cos(lat)
        else:
            w = (np.broadcast_to(np.expand_dims(weights,axis=1), np.shape(data)[2:4]))
            return data * w
    #usually you'll pass an xarray, but if you had a numpy array already, make sure it's shape is as expected (sample x lat x lon x channel)
    else:
        assert len(np.shape(data)) == 4, "excepted np.array of len(shape)==4)"
        weights = np.cos(np.deg2rad(lat))
        sum_weights = np.nansum(np.ones((data.shape[1], data.shape[2]))*weights[:, np.newaxis])
        global_mean = np.nansum(data*weights[np.newaxis, :, np.newaxis, np.newaxis], axis=(1, 2))/sum_weights

    return global_mean



