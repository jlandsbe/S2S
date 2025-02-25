"""Produce various model diagnostics, including plots and metrics.

"""
import build_data
import numpy as np
import time
import matplotlib.pyplot as plt
import plots
import regions
import metrics
import base_directories
from multiprocessing import Pool
import gc
import os
import warnings
from shapely.errors import ShapelyDeprecationWarning
import matplotlib.animation as animation
import scipy.stats
import random
import pickle
import CRPS
warnings.filterwarnings(action='ignore', message='Mean of empty slice')
warnings.filterwarnings(action='ignore', message='All-NaN slice encountered')
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
warnings.filterwarnings("ignore", message="Failed to load image Python extension")

__author__ = "Jacob Landsberg, Jamin K. Rader, Elizabeth A. Barnes, and Randal J. Barnes"
__version__ = "6 May 2024"

dir_settings = base_directories.get_directories()
dpiFig = 300




def visualize_metrics(settings, model, soi_input, soi_output, analog_input, analog_output, progression_analogs, progression_soi,
                      lat, lon, mask, persist_err=0, n_testing_analogs=1_000, n_testing_soi=1_000,
                      analogue_vector=None, fig_savename="", analog_dates = None, soi_dates = None,
                      soi_train_output=None, my_masks = None, gates = None, percentile = 0):
    if analogue_vector is None:
        analogue_vector = [1, 2, 5, 10, 15, 20, 25, 30]

    rng_eval = np.random.default_rng(seed=settings["rng_seed"]+5)

    # define analog_set size, either all data, or specified amount in experiments.py
    n_testing_analogs = np.min([analog_input.shape[0], n_testing_analogs])
    # get random analogs
    i_analog = rng_eval.choice(np.arange(0, analog_input.shape[0]), n_testing_analogs, replace=False)
    if percentile != 0:
        if percentile < 0:
            percentile_value = np.percentile(soi_output, -percentile)
            # Find the indices of soi_output that are less than or equal to the percentile value
            i_soi = np.where(soi_output <= percentile_value)[0]
        else:
            percentile_value = np.percentile(soi_output, percentile)
    # Find the indices of soi_output that are greater than or equal to the percentile value
            i_soi = np.where(soi_output >= percentile_value)[0]
    else:
        i_soi = rng_eval.choice(np.arange(0, soi_input.shape[0]), n_testing_soi, replace=False)
    if type(analog_dates) != type(None):
        analog_dates = analog_dates[i_analog]
    if type(soi_dates) != type(None):
        soi_dates = soi_dates[i_soi]
    if len(settings["progressions"])>0:
        prog_analogs = progression_analogs[:,i_analog,:,:,:]
        prog_soi = progression_soi[:,i_soi]
    else:
        prog_analogs = []
        prog_soi = []
    # assess model performance and compare to baselines
        metrics_dict, crps_dict = assess_metrics(settings, model,
                                  soi_input[i_soi, :, :, :],
                                  soi_output[i_soi],
                                  analog_input[i_analog, :, :, :],
                                  analog_output[i_analog],
                                  prog_analogs,
                                  prog_soi,
                                  lat, lon,
                                  mask, persist_err,
                                  soi_train_output=soi_train_output,
                                  analogue_vector=analogue_vector,
                                  fig_savename=fig_savename, my_masks = my_masks, gates = gates, analog_dates = analog_dates, soi_dates = soi_dates)

    return metrics_dict, crps_dict


def run_complex_operations(operation, inputs, pool, chunksize):
    return pool.imap(operation, inputs, chunksize=chunksize)


def soi_iterable(n_analogs, soi_input, soi_output, analog_input, analog_output, mask, best_analogs, uncertainties = 0, val_soi_output=None, val_analog_output=None, progression_analog=[], progression_soi=[],gates = None):
    """
    Create an iterable for a parallel approach to metric assessment
    """
    for i_soi in range(soi_input.shape[0]):
        if gates != None:
            mask = gates[i_soi]
        if type(val_soi_output) == type(None):
            val_soi_output = soi_output
            val_analog_output = analog_output
        if len(progression_analog)>1:
            inputs = {"n_analogs": n_analogs,
                    "max_analogs": np.max(n_analogs),
                    "analog_input": analog_input,
                    "analog_output": analog_output,
                    "soi_input_sample": soi_input[i_soi, :, :, :],
                    "soi_output_sample": soi_output[i_soi],
                    "mask": mask,
                    "uncertainties": uncertainties,
                    "val_soi_output": val_soi_output[i_soi],
                    "val_analog_output": val_analog_output,
                    "progression_soi": progression_soi[:,i_soi,:,:,:],
                    "progression_analog": progression_analog,
                    "best_analogs": best_analogs if best_analogs is None else best_analogs[i_soi]
                    }
        else:
            inputs = {"n_analogs": n_analogs,
                    "max_analogs": np.max(n_analogs),
                    "analog_input": analog_input,
                    "analog_output": analog_output,
                    "soi_input_sample": soi_input[i_soi, :, :, :],
                    "soi_output_sample": soi_output[i_soi],
                    "mask": mask,
                    "uncertainties": uncertainties,
                    "val_soi_output": val_soi_output[i_soi],
                    "val_analog_output": val_analog_output,
                    "progression_soi": progression_soi,
                    "progression_analog": progression_analog,
                    "best_analogs": best_analogs if best_analogs is None else best_analogs[i_soi]}
        yield inputs

def soi_iterable_dates(n_analogs, soi_input, soi_output, analog_input, analog_output, mask, analog_dates_months, soi_dates_months, analog_dates_years, soi_dates_years, uncertainties = 0, gates = None):
    """
    Create an iterable for a parallel approach to metric assessment
    """
    for i_soi in range(soi_input.shape[0]):
        if gates != None:
            mask = gates[i_soi]
        inputs = {"n_analogs": n_analogs,
                  "max_analogs": np.max(n_analogs),
                  "analog_input": analog_input,
                  "analog_output": analog_output,
                  "soi_input_sample": soi_input[i_soi, :, :, :],
                  "soi_output_sample": soi_output[i_soi],
                  "mask": mask,
                  "analog_months": analog_dates_months,
                  "soi_months": soi_dates_months,
                  "analog_years": analog_dates_years,
                  "soi_years": soi_dates_years,
                  "uncertainties": uncertainties,
                  }
        yield inputs

def soi_iterable_masks(n_analogs, soi_input, soi_output, analog_input, analog_output, masks):
    """
    Create an iterable for a parallel approach to metric assessment
    """
    for i_soi in range(soi_input.shape[0]):
        inputs = {"n_analogs": n_analogs,
                  "max_analogs": np.max(n_analogs),
                  "analog_input": analog_input,
                  "analog_output": analog_output,
                  "soi_input_sample": (soi_input[i_soi, :, :, :]),
                  "soi_output_sample": soi_output[i_soi],
                  "mask": masks[i_soi,:,:,np.newaxis],
                  }
        yield inputs


import matplotlib.pyplot as plt
import numpy as np
import random

def plot_histogram(selected_analogs_histogram, random_soi_output, dir_settings, settings, regional_analogs_histogram = None, all_analogs_histogram = None):
    plt.style.use("default")
    # Step 2: Set up histogram parameters (customize as needed)
    num_bins = 10  # Example number of bins
    if type(all_analogs_histogram)!=type(None):
        hist_range = (np.min(all_analogs_histogram), np.max(all_analogs_histogram))
    else:
        hist_range = (np.min(selected_analogs_histogram), np.max(selected_analogs_histogram))  # Range of histogram
    hist_color = 'deepskyblue'  # Color of the histogram bars
    hist_label = 'Analog Outputs'  # Label for the histogram

    # Step 3: Create the histogram
    plt.figure(figsize=(10, 6))  # Create a new figure for the histogram

    if type(regional_analogs_histogram)!=type(None):
        plt.hist(regional_analogs_histogram, bins=num_bins, range=hist_range, color='orchid', label='Regional Analog Outputs', alpha = 1, density=True, histtype='step', linewidth=8) 
    plt.hist(selected_analogs_histogram, bins=num_bins, range=hist_range, color=hist_color, label=hist_label, alpha =1, density=True,histtype='step', linewidth=8)
    if type(all_analogs_histogram)!=type(None):
        plt.hist(all_analogs_histogram, bins=num_bins, range=hist_range, color='moccasin', label='All Analog Outputs', alpha = .5, density=True, zorder=0)
    # Plot a vertical line at the value of random_soi_output
    plt.axvline(x=random_soi_output, color='midnightblue', linestyle='--', linewidth=2, label='Truth')
    plt.title('Histogram of Selected Analog Outputs')  # Step 4: Label the histogram
    plt.xlabel('Output Value (sigma)')
    plt.ylabel('Frequency')
    plt.legend()  # Add a legend to the plot
        # Remove top and right spines
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Step 5: Save the histogram
    histogram_filename = dir_settings["figure_directory"] + settings["savename_prefix"] + '_histogram.png'
    plt.savefig(histogram_filename, dpi=100, bbox_inches='tight')  # Save the figure
    plt.close()  # Close the figure to prevent it from displaying in the notebook or script output

def plot_bands(settings, dir_settings, total_max, total_min, truth, mask_min, mask_max, regional_min = None, regional_mask = None, x = None, filename_suffix = "range_bands.png", title_text = "?"):
    plt.style.use("default")
    if x == None:
        x = np.arange(len(truth))  # X-axis as index number

    plt.figure(figsize=(10, 6))
    
    # Plot the truth value as a black line
    plt.plot(x, truth, color='midnightblue', label='Truth', linewidth=3)
    
    # Plot the moccasin band between total_min and total_max
    plt.fill_between(x, total_min, total_max, color='moccasin', alpha=0.5, label='Total Range')
    if type(regional_mask) != type(None):
        plt.fill_between(x, regional_min, regional_mask, color='orchid', alpha=0.5, label='Regional Range', edgecolor='orchid', linewidth=1.5)
    # Plot the deepskyblue band between mask_min and mask_max
    plt.fill_between(x, mask_min, mask_max, color='deepskyblue', alpha=0.5, label='Mask Range', edgecolor='deepskyblue', linewidth=1.5)

    
    # Set labels and title
    plt.xlabel('Index Number')
    plt.ylabel('Output Value (sigma)')
    plt.title('Confinement of Analog Outputs for ' + title_text + " Analogs")
    plt.legend()
    
    # Remove top and right spines
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    z = .08*(np.max(total_max) - np.min(total_min))
    plt.ylim(np.min(total_min) - z, np.max(total_max) + z)
    
        # Save the plot to a file
    plt.savefig(dir_settings["figure_directory"] + settings["savename_prefix"] + filename_suffix, dpi=100, bbox_inches='tight')
    plt.close()

def create_subplots(random_soi_input, random_soi_output, selected_analogs, selected_analogs_output, mask, settings, lat, lon, dir_settings, filename_suffix):
    plt.style.use("default")
    if random_soi_input.shape[-1] != 1:
        fig, axs = plt.subplots(3, 6, figsize=(24, 12))
        fig.tight_layout()
        soi_val = str(round(random_soi_output, 2))
        # First plot: random_soi_input
        for k in range(2):
            ax1, climits = plots.plot_interp_masks(
                fig=fig,
                settings=settings,
                weights_train=np.squeeze(random_soi_input[:,:,k]),
                lat=lat,
                lon=lon,
                title_text=f"Truth: {soi_val}",
                subplot=(3, 6, k+1),
                use_text=0,
                cbarBool=False  # Disable individual colorbars
            )

        predicted_val = str(round(np.mean(selected_analogs_output), 2))

        # Subsequent plots: selected_analogs
        for i in range(8):
            for j in range(2):
                title_val = str(round(selected_analogs_output[i], 2))
                ax, _ = plots.plot_interp_masks(
                    fig=fig,
                    settings=settings,
                    weights_train=np.squeeze(selected_analogs[i,:,:,j]),
                    lat=lat,
                    lon=lon,
                    title_text=f"Predicts: {title_val}",
                    subplot=(3, 6, 2*i + j + 3),
                    use_text=0,
                    cbarBool=False  # Disable individual colorbars
                )
    else:
        fig, axs = plt.subplots(3, 3, figsize=(12, 8))
        fig.tight_layout()
        soi_val = str(round(random_soi_output, 2))
        # First plot: random_soi_input
        ax1, climits = plots.plot_interp_masks(
            fig=fig,
            settings=settings,
            weights_train=np.squeeze(random_soi_input),
            lat=lat,
            lon=lon,
            title_text=f"Truth: {soi_val}",
            subplot=(3, 3, 1),
            use_text=0,
            cbarBool=False  # Disable individual colorbars
        )

        predicted_val = str(round(np.mean(selected_analogs_output), 2))

        # Subsequent plots: selected_analogs
        for i in range(8):
            title_val = str(round(selected_analogs_output[i], 2))
            ax, _ = plots.plot_interp_masks(
                fig=fig,
                settings=settings,
                weights_train=np.squeeze(selected_analogs[i]),
                lat=lat,
                lon=lon,
                title_text=f"Predicts: {title_val}",
                subplot=(3, 3, i + 2),
                use_text=0,
                cbarBool=False  # Disable individual colorbars
            )

    # Turn off all spines for each subplot
    for ax in axs.flat:
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Add a single colorbar for the entire figure
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap=plots.get_mycolormap(), norm=plt.Normalize(vmin=climits[0], vmax=climits[1])),
        ax=axs,  # Reference to the entire grid of subplots
        orientation='horizontal',  # Choose 'horizontal' or 'vertical'
        fraction=0.02,  # Fraction of the plot occupied by the colorbar
        pad=0.1  # Padding between the plot and colorbar
    )

    for ax in axs.flat:  # Iterate over each subplot (axes)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    fig.text(0.5, 0.99, f"Predicted Value: {predicted_val}", ha='center', fontweight='bold')
    # Save the figure
    fig.savefig(dir_settings["figure_directory"] + settings["savename_prefix"] + filename_suffix, dpi=dpiFig, bbox_inches='tight')
    plt.close()

def process_results(run_complex_operations, metrics_function, soi_iterable, pool, n_analogues, 
                    soi_input_shape, metrics_directory, savename_prefix, file_suffix, chunksize=1000):
    """
    Runs the complex operations and processes the results, including appending errors and best analogs,
    saving the best analogs to a file, and returning error metrics.

    Args:
        run_complex_operations: Function to run operations.
        metrics_function: Metrics function (e.g., mse_operation) used for the complex operations.
        soi_iterable: Iterable, data to be passed to the complex operations.
        pool: Multiprocessing pool for parallel operations.
        n_analogues: List, number of analogues.
        soi_input_shape: Tuple, shape of the input data for initializing arrays.
        metrics_directory: String, path to directory where metrics will be saved.
        savename_prefix: String, prefix for saving the pickle file.
        file_suffix: String, suffix for the file name to distinguish between local/global/etc.
        chunksize: Integer, the chunk size for multiprocessing.

    Returns:
        Tuple of error metrics (error_network, analog_match_error, prediction_spread, 
        prediction_IQR, prediction_min, prediction_max, prediction_range).
    """
    err = []
    all_best_analogs = np.zeros((soi_input_shape[0], n_analogues[-1]))
    soi_idx = 0

    # Running complex operations and processing results
    for result in run_complex_operations(metrics_function, soi_iterable, pool, chunksize=chunksize):
        errors = result[:-1]  # Grab all metrics besides the best analogs
        best_analogs = result[-1]  # Grab the best analogs
        err.append(errors)
        all_best_analogs[soi_idx] = np.array(best_analogs)
        soi_idx += 1

    # Save best analogs to a pickle file
    save_path = f"{metrics_directory}{savename_prefix}_best_{file_suffix}_analogs.pickle"
    with open(save_path, 'wb') as f:
        pickle.dump(all_best_analogs.astype(int), f)

    # Convert net_err to numpy array for further processing
    net_err = np.array(err)  # shape soi x metric # x n_analogs

    # Extract various error metrics
    error_network = net_err[:, 0, :]
    analog_match_error = net_err[:, 1, :]
    prediction_spread = net_err[:, 2, :]
    prediction_IQR = net_err[:, 3, :]
    prediction_min = net_err[:, 4, :]
    prediction_max = net_err[:, 5, :]
    prediction_value = net_err[:, 6, :]
    prediction_CRPS = net_err[:, 7, :]
    prediction_range = prediction_max - prediction_min

    return error_network, analog_match_error, prediction_spread, prediction_IQR, prediction_min, prediction_max, prediction_range, prediction_value, prediction_CRPS

def assess_metrics(settings, model, soi_input, soi_output, analog_input,
                   analog_output, progression_analog, progression_soi, lat, lon,
                   mask, persist_err=0,
                   soi_train_output=None,
                   analogue_vector=[1, 2, 5, 10, 15, 20, 25, 30],
                   show_figure=False, save_figure=True, fig_savename="", my_masks=None, gates = None, analog_dates = None, soi_dates = None):
    ignore_baselines = 1
    if settings["median"]:
        analog_output_val = analog_output
        soi_output_val = soi_output
        soi_output = 1.0*(soi_output > np.median(soi_output))
        analog_output = 1.0*(analog_output > np.median(analog_output))


    if settings["percentiles"]!=None:
        analog_output_val = analog_output
        soi_output_val = soi_output
        low_cap_an = np.percentile(analog_output, settings["percentiles"][0], axis = 0)
        high_base_an = np.percentile(analog_output, settings["percentiles"][1], axis = 0)
        low_cap_soi = np.percentile(soi_output, settings["percentiles"][0], axis = 0)
        high_base_soi = np.percentile(soi_output, settings["percentiles"][1], axis = 0)
        analog_output = np.where(analog_output <= low_cap_an, -1, np.where(analog_output>=high_base_an, 1, 0))
        soi_output = np.where(soi_output <= low_cap_soi, -1, np.where(soi_output>=high_base_soi, 1, 0))
    else:
        analog_output_val = None
        soi_output_val = None

    # Number of Processes for Pool (all but two)
    n_processes = os.cpu_count() - 2

    # Create RNG
    rng = np.random.default_rng(settings["rng_seed"])

    # Determine all the number of analogs to assess
    if settings["model_type"] == "ann_model":
        analogue_vector = [15,] # Compare ANN_MODEL results to other baselines using 15 analogs
    len_analogues = len(analogue_vector)

    # These ones require parallelization, and must be transposed
    error_network = np.zeros((len_analogues, soi_input.shape[0])).T * np.nan
    error_corr = np.zeros((len_analogues, soi_input.shape[0])).T * np.nan
    error_customcorr = np.zeros((len_analogues, soi_input.shape[0])).T * np.nan
    error_globalcorr = np.zeros((len_analogues, soi_input.shape[0])).T * np.nan
    global_crps = np.zeros((len_analogues, soi_input.shape[0])).T * np.nan
    NH_crps = np.zeros((len_analogues, soi_input.shape[0])).T * np.nan

    # These ones do not require parallelization
    if settings["error_calc"]!="map":
        error_random = np.zeros((len_analogues, soi_input.shape[0])) * np.nan
        error_climo = np.zeros((soi_input.shape[0])) * np.nan
    #error_persist = np.zeros((soi_input.shape[0] - \len(settings["soi_test_members"]) * np.abs(settings["smooth_len_output"]))) * np.nan
        error_custom = np.zeros((soi_input.shape[0] - \
                              len(settings["soi_test_members"]) * np.abs(settings["smooth_len_output"]))) * np.nan
    else:
        error_random = []
        error_climo = []
        error_custom = []

    n_analogues = analogue_vector


    print("calculating metrics for all analogs: " + str(n_analogues) )
    print(np.shape(analog_input))
    time_start = time.perf_counter()

    # -----------------------
    # Interpretable-Analog
    if settings["model_type"] == "interp_model":
            
            if mask is None:
                lst = []
            #     if settings["error_calc"] == "map":
            #         error_network = np.array([])
            #     #for i_analogue_loop, nlog in enumerate(analogue_vector):
            # # let the network tell us every prediction what mask to use
            # # this code is very slow, but the memory leak has been dealt with (xarrray did not have this issue)
            #     for nlog in analogue_vector:
            #         temp_hold = []
            #         for sample in np.arange(0, soi_input.shape[0]):

            #             soi_input_sample = soi_input[sample, :, :, :]
            #             soi_output_sample = soi_output[sample]

            #             prediction_test = model.predict(
            #                 [np.broadcast_to(soi_input_sample,
            #                                 (analog_input.shape[0],
            #                                 soi_input_sample.shape[0],
            #                                 soi_input_sample.shape[1],
            #                                 soi_input_sample.shape[2])
            #                                 ),
            #                 analog_input],
            #                 batch_size=10_000,
            #             )
            #             # this gc.collect must be included or there is a major memory leak when model.predict is in a loop
            #             # https://stackoverflow.com/questions/64199384/tf-keras-model-predict-results-in-memory-leak
            #             # https://github.com/tensorflow/tensorflow/issues/44711
            #             _ = gc.collect()
            #             i_analogue = np.argsort(prediction_test, axis=0)[:nlog, 0]
            #             if settings["error_calc"] == "field":
            #                 temp_hold.append(metrics.get_analog_errors(soi_output_sample, np.mean(analog_output[i_analogue], axis=0), settings["error_calc"]))
            #             else:
            #                 error_network[i_analogue_loop, sample] = metrics.get_analog_errors(soi_output_sample,
            #                                                                             np.mean(analog_output[i_analogue]), settings["error_calc"])
                #     lst.append(temp_hold)
                # error_network = np.swapaxes(np.array(lst),0,1)  
                # #error_network = np.swapaxes(error_network,0,1)
            else:
                with Pool(n_processes) as pool:
                    
                    if type(analog_dates) !=type(None):
                            analog_months = np.array([date.month for date in analog_dates])
                            analog_years = np.array([date.year for date in analog_dates])
                            soi_months = np.array([date.month for date in soi_dates])
                            soi_years = np.array([date.year for date in soi_dates])
                            soi_iterable_instance = soi_iterable_dates(n_analogues,
                                                            soi_input,
                                                            soi_output,
                                                            analog_input,
                                                            analog_output,
                                                            mask, analog_months, soi_months, analog_years, soi_years)
                            date_info = np.squeeze(np.array(run_complex_operations(metrics.date_operation,
                                                                    soi_iterable_instance,
                                                                    pool,
                                                                    chunksize=soi_input.shape[0]//n_processes,)))
                            month_info = date_info[:,0,:]
                            year_info = date_info[:,1,:]
                            #Date info plotting
                            year_length = max(analog_years) - min(analog_years)
                            soi_months_repeated =  np.repeat(soi_months,month_info.shape[1])
                            best_analog_months = month_info.flatten()
                            soi_year_repeated =  np.repeat((soi_years),year_info.shape[1])
                            best_analog_years= year_info.flatten()
                            plots.yearly_analysis(soi_year_repeated, best_analog_years, year_length, settings)
                            plots.monthly_analysis(soi_months_repeated, best_analog_months, settings)
                            exit()
                    else:
                        best_analogs_path = dir_settings["metrics_directory"]+settings["savename_prefix"] + '_best_masked_analogs.pickle'
                        if not os.path.exists(best_analogs_path):
                            # If we don't have analogs saved, we will calculate them later
                            net_best_analogs = None
                        else:
                            # If we do already have them saved, we will load them
                            # Load the array from the pickle file
                            with open(best_analogs_path, 'rb') as file:
                                net_best_analogs = pickle.load(file)
                        soi_iterable_instance = soi_iterable(n_analogues,
                                                            soi_input,
                                                            soi_output,
                                                            analog_input,
                                                            analog_output,
                                                            mask, net_best_analogs, 1, val_analog_output=analog_output_val,val_soi_output=soi_output_val, progression_analog=progression_analog, progression_soi=progression_soi, gates = gates)
                        

 
                    if (settings["median"] or settings["percentiles"]!=None):
                        # net_err = np.array(run_complex_operations(metrics.super_classification_operation,
                        #                                             soi_iterable_instance,
                        #                                             pool,
                        #                                             chunksize=soi_input.shape[0]//n_processes,))
                        # error_network[:, :] = net_err[:,0,:]
                        # analog_match_error = net_err[:,1,:] 
                        # prediction_spread = net_err[:,2,:]
                        # modal_fraction = net_err[:,3,:]
                        # entropy_spread = net_err[:,4,:]
                        error_network, analog_match_error, prediction_spread, prediction_entropy, prediction_min, prediction_max, prediction_range, predicted_val, prediction_crps = process_results(
                            run_complex_operations, metrics.super_classification_operation, soi_iterable_instance, pool, 
                            n_analogues, soi_input.shape, dir_settings["metrics_directory"], 
                            settings["savename_prefix"], "masked"
                        )
        
                    elif settings["error_calc"] == "classify":
                        error_network[:, :] = run_complex_operations(metrics.classification_operation,
                                                                    soi_iterable_instance,
                                                                    pool,
                                                                    chunksize=soi_input.shape[0]//n_processes,)
                    elif settings["error_calc"] == "map":
                        #number of analogs x lat x lon
                        error_network, analog_match_error, prediction_spread,prediction_IQR, prediction_min, prediction_max,prediction_range, predicted_val, prediction_crps = process_results(
                            run_complex_operations, metrics.map_operation, soi_iterable_instance, pool, 
                            n_analogues, soi_input.shape, dir_settings["metrics_directory"], 
                            settings["savename_prefix"], "masked"
                        )
                    elif settings["error_calc"] == "field":
                        net_err = np.array(run_complex_operations(metrics.field_operation,
                                                                    soi_iterable_instance,
                                                                    pool,
                                                                    chunksize=soi_input.shape[0]//n_processes,))

                    else:
                        
                        error_network, analog_match_error, prediction_spread,prediction_IQR, prediction_min, prediction_max, prediction_range, predicted_val, prediction_crps = process_results(
                            run_complex_operations, metrics.mse_operation, soi_iterable_instance, pool, 
                            n_analogues, soi_input.shape, dir_settings["metrics_directory"], 
                            settings["savename_prefix"], "masked"
                        )










                    print("finished network error")
    # -----------------------
    # ANN-Analog
    elif settings["model_type"] == "ann_analog_model":
        # let the network tell us every prediction what mask to use
        # this code is very slow, but the memory leak has been dealt with (xarray did not have this issue)
        for sample in np.arange(0, soi_input.shape[0]):

            soi_input_sample = soi_input[sample, :, :, :]
            soi_output_sample = soi_output[sample]

            prediction_test = model.predict(
                [np.broadcast_to(soi_input_sample,
                                    (analog_input.shape[0],
                                    soi_input_sample.shape[0],
                                    soi_input_sample.shape[1],
                                    soi_input_sample.shape[2])
                                    ),
                    analog_input],
                batch_size=10_000,
            )
            # this gc.collect must be included or there is a major memory leak when model.predict is in a loop
            # https://stackoverflow.com/questions/64199384/tf-keras-model-predict-results-in-memory-leak
            # https://github.com/tensorflow/tensorflow/issues/44711
            _ = gc.collect()
            i_analogues = np.argsort(prediction_test, axis=0)
            for idx_analog, n_analog in enumerate(n_analogues):
                i_analogue = i_analogues[:n_analog, 0]
                if settings["error_calc"] == "super_classify":
                    error_network[sample, idx_analog] = metrics.get_analog_errors(soi_output_sample,
                                                                    np.median(analog_output[i_analogue]), settings["error_calc"]) #
                else:
                    error_network[sample, idx_analog] = metrics.get_analog_errors(soi_output_sample,
                                                                    np.mean(analog_output[i_analogue], axis=0), settings["error_calc"])

    # -----------------------
    # Vanilla ANN Model (only need to compute once, not a function of n_analogues)
    elif settings["model_type"] == "ann_model":
        for sample in np.arange(0, soi_input.shape[0]):
            soi_input_sample = soi_input[sample, :, :, :]
            soi_output_sample = soi_output[sample]
            prediction_test = model.predict([soi_input_sample[np.newaxis, :, :, :],
                                                soi_input_sample[np.newaxis, :, :, :]])
            _ = gc.collect()
            error_network[sample, :] = metrics.get_analog_errors(soi_output_sample, prediction_test, settings["error_calc"]).T

    # -----------------------
    # Simple GLOBAL correlation baseline
    if not ignore_baselines or 1:
        with Pool(n_processes) as pool:
            sqrt_area_weights = np.sqrt(np.abs(np.cos(np.deg2rad(lat)))[np.newaxis, :, np.newaxis, np.newaxis])
            best_global_analogs_path = dir_settings["metrics_directory"]+settings["savename_prefix"] + '_best_global_analogs.pickle'
            if not os.path.exists(best_global_analogs_path):
                # If we don't have analogs saved, we will calculate them later
                best_global_analogs = None
            else:
                # If we do already have them saved, we will load them
                # Load the array from the pickle file
                with open(best_global_analogs_path, 'rb') as file:
                    best_global_analogs = pickle.load(file)
            soi_iterable_instance = soi_iterable(n_analogues,
                                            soi_input,
                                            soi_output,
                                            analog_input,
                                            analog_output,
                                            sqrt_area_weights, best_global_analogs, uncertainties=1, val_analog_output=analog_output_val, val_soi_output=soi_output_val, progression_analog=progression_analog, progression_soi=progression_soi)
            if (settings["median"] or settings["percentiles"]!=None):

                error_globalcorr, global_analog_match_error, global_prediction_spread, \
                global_entropy, global_min, global_max, \
                global_range, global_predicted_val, global_crps = process_results(
                run_complex_operations, metrics.super_classification_operation, soi_iterable_instance, pool, 
                n_analogues, soi_input.shape, dir_settings["metrics_directory"], 
                settings["savename_prefix"], "global")
            elif settings["error_calc"] == "classify":
                error_globalcorr[:, :] = run_complex_operations(metrics.classification_operation,
                                                                soi_iterable_instance,
                                                                pool,
                                                                chunksize=soi_input.shape[0]//n_processes,)
            elif settings["error_calc"] == "map":
                error_globalcorr, global_analog_match_error, global_prediction_spread, \
                global_IQR, global_min, global_max, \
                global_range, global_predicted_val, global_crps = process_results(
                run_complex_operations, metrics.map_operation, soi_iterable_instance, pool, 
                n_analogues, soi_input.shape, dir_settings["metrics_directory"], 
                settings["savename_prefix"], "global")
            elif settings["error_calc"] == "field":
                error_globalcorr[:, :] = run_complex_operations(metrics.field_operation,
                                                                soi_iterable_instance,
                                                                pool,
                                                                chunksize=soi_input.shape[0]//n_processes,)
            else:
                error_globalcorr, global_analog_match_error, global_prediction_spread, \
                global_IQR, global_min, global_max, \
                global_range, global_predicted_val, global_crps = process_results(
                run_complex_operations, metrics.mse_operation, soi_iterable_instance, pool, 
                n_analogues, soi_input.shape, dir_settings["metrics_directory"], 
                settings["savename_prefix"], "global")
            print("finished global error")
    # -----------------------
    # Simple TARGET REGION correlation baseline
    with Pool(n_processes) as pool:
        soi_reg, lat_reg, lon_reg = build_data.extract_region(soi_input, regions.get_region_dict(
            settings["target_region_name"]), lat=lat, lon=lon)
        analog_reg, __, __ = build_data.extract_region(analog_input,
                                                        regions.get_region_dict(settings["target_region_name"]),
                                                        lat=lat, lon=lon)
        if len(settings["progressions"])>0:
            progression_analog_reg = build_data.extract_region(progression_analog,
                                                        regions.get_region_dict(settings["target_region_name"]),
                                                        lat=lat, lon=lon, progressions=1)
            progression_soi_reg = build_data.extract_region(progression_soi,
                                            regions.get_region_dict(settings["target_region_name"]),
                                            lat=lat, lon=lon, progressions=1)
        else:
            progression_analog_reg = []
            progression_soi_reg = []
        sqrt_area_weights = np.sqrt(np.abs((np.cos(np.deg2rad(lat_reg))[np.newaxis, :, np.newaxis, np.newaxis])))
        best_region_analogs_path = dir_settings["metrics_directory"]+settings["savename_prefix"] + '_best_region_analogs.pickle'
        if not os.path.exists(best_region_analogs_path):
            # If we don't have analogs saved, we will calculate them later
            best_region_analogs = None
        else:
            # If we do already have them saved, we will load them
            # Load the array from the pickle file
            with open(best_region_analogs_path, 'rb') as file:
                best_region_analogs = pickle.load(file)
        soi_iterable_instance = soi_iterable(n_analogues,
                                                soi_reg,
                                                soi_output,
                                                analog_reg,
                                                analog_output,
                                                sqrt_area_weights, best_region_analogs,uncertainties = 1, val_analog_output=analog_output_val, val_soi_output=soi_output_val, progression_analog=progression_analog_reg, progression_soi=progression_soi_reg)
        if (settings["median"] or settings["percentiles"]!=None):
            error_corr, regional_analog_match_error, regional_prediction_spread, \
            regional_entropy, regional_min, regional_max, \
            regional_range, regional_predicted_val, regional_crps = process_results(
            run_complex_operations, metrics.super_classification_operation, soi_iterable_instance, pool, 
            n_analogues, soi_input.shape, dir_settings["metrics_directory"], 
            settings["savename_prefix"], "regional")
        elif settings["error_calc"] == "map":
            error_corr, regional_analog_match_error, regional_prediction_spread, \
            regional_IQR, regional_min, regional_max, \
            regional_range, regional_predicted_val, regional_crps = process_results(
            run_complex_operations, metrics.map_operation, soi_iterable_instance, pool, 
            n_analogues, soi_input.shape, dir_settings["metrics_directory"], 
            settings["savename_prefix"], "regional")
        elif settings["error_calc"] == "classify":
            error_corr[:, :] = run_complex_operations(metrics.classification_operation,
                                                            soi_iterable_instance,
                                                            pool,
                                                            chunksize=soi_input.shape[0]//n_processes,)
        elif settings["error_calc"] == "field":
            error_corr[:, :] = run_complex_operations(metrics.field_operation,
                                                            soi_iterable_instance,
                                                            pool,
                                                            chunksize=soi_input.shape[0]//n_processes,)
        else:
            error_corr, regional_analog_match_error, regional_prediction_spread, \
            regional_IQR, regional_min, regional_max, \
            regional_range, regional_predicted_val, regional_crps = process_results(
            run_complex_operations, metrics.mse_operation, soi_iterable_instance, pool, 
            n_analogues, soi_input.shape, dir_settings["metrics_directory"], 
            settings["savename_prefix"], "regional")
            
        print("finished target region error")
        
    # -----------------------
    # Simple CUSTOM CORRELATION REGION correlation baseline (not needed)
    if not ignore_baselines and "correlation_region_name" in settings.keys():
        with Pool(n_processes) as pool:
            cust_reg_map = np.zeros(np.shape(mask))
            cust_reg_map = build_data.extract_region(cust_reg_map, regions.get_region_dict(settings["correlation_region_name"]), lat=lat, lon=lon, mask_builder = 1)
            best_cust_analogs_path = dir_settings["metrics_directory"]+settings["savename_prefix"] + '_best_cust_analogs.pickle'
            if not os.path.exists(best_cust_analogs_path):
            # If we don't have analogs saved, we will calculate them later
                best_cust_analogs = None
            else:
            # If we do already have them saved, we will load them
            # Load the array from the pickle file
                with open(best_cust_analogs_path, 'rb') as file:
                    best_cust_analogs = pickle.load(file)
            if (settings["median"] or settings["error_calc"]=="mse" or settings["percentiles"]!=None):
                soi_iterable_instance = soi_iterable(n_analogues,
                                                soi_input,
                                                soi_output,
                                                analog_input,
                                                analog_output,
                                                cust_reg_map, best_cust_analogs, uncertainties = 1, val_analog_output=analog_output_val, val_soi_output=soi_output_val, progression_analog=progression_analog, progression_soi=progression_soi)
            else:
                soi_iterable_instance = soi_iterable(n_analogues,
                                                soi_input,
                                                soi_output,
                                                analog_input,
                                                analog_output,
                                                cust_reg_map, best_cust_analogs, progression_analog=progression_analog, progression_soi=progression_soi)
            if (settings["median"] or settings["percentiles"]!=None):
                error_customcorr, NH_analog_match_error, NH_prediction_spread, \
                NH_entropy, NH_min, NH_max, \
                NH_range, NH_predicted_val, NH_crps = process_results(
                run_complex_operations, metrics.super_classification_operation, soi_iterable_instance, pool, 
                n_analogues, soi_input.shape, dir_settings["metrics_directory"], 
                settings["savename_prefix"], "global")
            
            elif settings["error_calc"] == "classify":
                error_customcorr[:, :] = run_complex_operations(metrics.classification_operation,
                                                            soi_iterable_instance,
                                                            pool,
                                                            chunksize=soi_input.shape[0]//n_processes,)
            elif settings["error_calc"] == "map":
                #number of analogs x time x lat x lon
                error_customcorr = run_complex_operations(metrics.map_operation,
                                                            soi_iterable_instance,
                                                            pool,
                                                            chunksize=soi_input.shape[0]//n_processes,)
            elif settings["error_calc"] == "field":
                error_customcorr[:, :] = run_complex_operations(metrics.field_operation,
                                                            soi_iterable_instance,
                                                            pool,
                                                            chunksize=soi_input.shape[0]//n_processes,)
            else:
                error_customcorr, NH_analog_match_error, NH_prediction_spread, \
                NH_IQR, NH_min, NH_max, \
                NH_range, NH_predicted_val, NH_crps = process_results(
                run_complex_operations, metrics.mse_operation, soi_iterable_instance, pool, 
                n_analogues, soi_input.shape, dir_settings["metrics_directory"], 
                settings["savename_prefix"], "global")
            print("finished custom region error")
    
    # -----------------------
    # Custom baseline (e.g. mean evolution)
    if "custom_baseline" in settings.keys() and not ignore_baselines:
        custom_true, custom_pred = metrics.calc_custom_baseline(settings["custom_baseline"], 
                                                                soi_output=soi_output,
                                                                soi_train_output=soi_train_output,
                                                                settings=settings)
        error_custom[:] = metrics.get_analog_errors(custom_true, custom_pred)

    # -----------------------
    # Random baseline
    if not ignore_baselines:
        random_output_spread = np.zeros((len_analogues, soi_input.shape[0])) * np.nan
        for idx_analog, n_analog in enumerate(n_analogues):
            i_analogue = rng.choice(np.arange(0, analog_output.shape[0]),
                                    size=(n_analog, soi_output.shape[0]), replace=True)
            if (settings["median"] or settings["percentiles"]!=None):
                error_random[idx_analog, :] = np.mean((soi_output!=(scipy.stats.mode(analog_output[i_analogue], axis=0)).mode))
                
            elif settings["error_calc"] == "map":
                np.append(error_random,metrics.get_analog_errors(soi_output,
                                                        np.median(analog_output[i_analogue], axis=0), settings["error_calc"]))
            else: 
                error_random[idx_analog, :] = metrics.get_analog_errors(soi_output,
                                                        np.mean(analog_output[i_analogue], axis=0), settings["error_calc"])
                if len(np.shape(analog_output)) > 2:
                    random_output_spread[idx_analog,:] = (np.mean(np.var(analog_output[i_analogue],axis=0), axis=(-1,-2)))
                else:
                    random_output_spread[idx_analog,:] = (np.var(analog_output[i_analogue], axis=0))


    # plots.uncertainty_whiskers(analogue_vector, error_network, analog_match_error, prediction_spread, settings, 
    #                                     baseline_error = [error_customcorr, error_globalcorr], baseline_analog_match = [NH_analog_match_error, global_analog_match_error], 
    #                                     baseline_spread = [NH_prediction_spread, global_prediction_spread], random_error = np.array(error_random).T, random_spread = random_output_spread.T)

    # -----------------------
    # Climatology
    error_climo_crps = None
    if (settings['median'] or settings["percentiles"]!=None) and settings["error_calc"] != "map":
        error_climo = np.repeat(np.array([1]), len_analogues)
        if settings["median"]:
            error_climo = np.ones(len(soi_output)) * .5
            error_climo_crps = np.ones(len(soi_output)) *  1/2#set to brier score of .5
        if settings["percentiles"]!=None:
            error_climo = np.ones(len(soi_output)) * 1-(1/(len(settings["percentiles"])+1))
            error_climo_crps = np.ones(len(soi_output)) * 2/3
    elif settings["error_calc"] == "map":
        if settings["median"]:
            error_climo = np.ones_like(soi_output) * .5
            error_climo_crps = np.ones_like(soi_output) *  1/2
        if settings["percentiles"]!=None:
            error_climo = np.ones_like(soi_output) * 1-(1/(len(settings["percentiles"])+1))
            error_climo_crps = np.ones_like(soi_output)* 2/3
        else:
        # Calculate climatological error (mean over analogs)
            error_climo = metrics.get_analog_errors(soi_output, np.mean(analog_output, axis=0), settings["error_calc"])
            
            # Initialize CRPS grid with NaNs
            error_climo_crps = np.full_like(error_climo, np.nan)
            
            
            # Loop through each sample/time index
            for idx, soi_output_ind in enumerate(soi_output):
                error_climo_crps[idx] = np.array([
                    [CRPS.CRPS(analog_output[:, lat, lon], soi_output_ind[lat, lon]).compute()[0]
                    for lon in range(error_climo_crps.shape[2])]  # assuming lat/lon are the 1st and 2nd dimensions
                    for lat in range(error_climo_crps.shape[1])
                ])
    else:
        error_climo_crps = np.zeros((soi_input.shape[0])) * np.nan
        for idx, soi_output_ind in enumerate(soi_output):
            error_climo_crps[idx] = (CRPS.CRPS(analog_output, soi_output_ind).compute()[0])
        error_climo[:] = metrics.get_analog_errors(soi_output, np.mean(analog_output, axis=0), settings["error_calc"]).T


    if settings["error_calc"] != "map":
    # -----------------------
        # Case Study Plots
        # Main script
        
        num_ans_hist = 30
        random.seed(21)
        length_of_soi_output = len(soi_output)
        random_index = random.randint(0, length_of_soi_output - 1)
        random_soi_output = soi_output[random_index]
        random_soi_input = soi_input[random_index]

        #find n closest matches' outputs based on mask
        mimse_mask = np.mean((random_soi_input * mask - analog_input * mask) ** 2, axis=(1, 2, 3))
        i_analogs_histogram_mask = np.argsort(mimse_mask, axis=0)[:num_ans_hist]
        selected_analogs_histogram_mask = analog_output[i_analogs_histogram_mask]
        #find n closest matches' outputs based on regional mask
        hist_soi_reg, lat_reg, lon_reg = build_data.extract_region(random_soi_input[np.newaxis, :], regions.get_region_dict(
                settings["target_region_name"]), lat=lat, lon=lon)
        mimse_regional = np.mean((hist_soi_reg - analog_reg) ** 2, axis=(1, 2, 3)) #here anlog_reg was defined earlier to be the regionally masked analogs
        i_analogs_histogram_regional = np.argsort(mimse_regional, axis=0)[:num_ans_hist]
        selected_analogs_histogram_regional = analog_output[i_analogs_histogram_regional]
        #all possible analog outputs
        selected_analogs_histogram_all = analog_output
        if len(np.shape(selected_analogs_histogram_mask)) > 1:
            selected_analogs_histogram_mask = np.mean(selected_analogs_histogram_mask, axis=(1, 2))
            selected_analogs_histogram_regional = np.mean(selected_analogs_histogram_regional, axis=(1, 2))
            selected_analogs_histogram_all = np.mean(selected_analogs_histogram_all, axis=(1, 2))
        # Plot and save histogram
        plot_histogram(selected_analogs_histogram_mask, random_soi_output, dir_settings, settings, selected_analogs_histogram_regional,selected_analogs_histogram_all)

        i_analogs = np.argsort(mimse_mask, axis=0)[:8]
        selected_analogs = analog_input[i_analogs]
        selected_analogs_output = analog_output[i_analogs]

        # Create and save subplots
        create_subplots(random_soi_input, random_soi_output, selected_analogs, selected_analogs_output, mask, settings, lat, lon, dir_settings, '_example.png')
        create_subplots(random_soi_input * mask, random_soi_output, selected_analogs * mask, selected_analogs_output, mask, settings, lat, lon, dir_settings, '_example_masked.png')

    #-----------------------
        #spread confinment plots
        all_outputs_max = np.mean(np.max(analog_output, axis=0))
        all_outputs_min = np.mean(np.min(analog_output, axis=0))
        num_soi_outputs = len(soi_output)
        num_ans_idx = 3
        plot_bands(settings, dir_settings, np.tile(all_outputs_min, num_soi_outputs), np.tile(all_outputs_max, num_soi_outputs),soi_output,prediction_min[:,num_ans_idx], prediction_max[:,num_ans_idx],regional_min[:,num_ans_idx], regional_max[:,num_ans_idx],title_text=str(settings["analogue_vec"][num_ans_idx]))


    # -----------------------
        #Confidence Plots
        if (settings["median"] or settings["percentiles"]!=None):
            network_confidence_dict = {"Ensemble Agreement": prediction_spread,}
            global_confidence_dict = {"Ensemble Agreement": global_prediction_spread,}
            NH_cofidence_dict = {}
            regional_cofidence_dict = {}
            random_confidence_dict = {}
            climatol = error_climo
        elif not ignore_baselines:
            network_confidence_dict = {"Analog Match": analog_match_error, "Prediction Spread": prediction_spread, "Prediction IQR": prediction_IQR, "Prediction Range": prediction_range}
            global_confidence_dict = {"Analog Match": global_analog_match_error, "Prediction Spread": global_prediction_spread, "Prediction IQR": global_IQR, "Prediction Range": global_range}
            NH_cofidence_dict = {"Analog Match": NH_analog_match_error, "Prediction Spread": NH_prediction_spread}
            regional_cofidence_dict = {"Analog Match": regional_analog_match_error, "Prediction Spread": regional_prediction_spread, "Prediction IQR": regional_IQR, "Prediction Range": regional_range}
            random_confidence_dict = {"Prediction Spread": random_output_spread.T}
            climatol = error_climo
        else:
            network_confidence_dict = {'Predicted Extremity': predicted_val,}
            global_confidence_dict = {}
            NH_cofidence_dict = {}
            regional_cofidence_dict = { 'Predicted Extremity': regional_predicted_val,}
            random_confidence_dict = {}
            climatol = error_climo

        
        #drop global for now:
        #global_confidence_dict={}
        NH_cofidence_dict={}
        #regional_cofidence_dict={}
        #global_confidence_dict={}
        net_col = "#2A9D8F"
        global_col = "#F4A261"
        regional_col = "#E76F51"
        error_conf_dict = {"Network":(error_network, network_confidence_dict, "solid",net_col), "Northern Hemisphere":(error_customcorr, NH_cofidence_dict, "dashed", "#E9C46A"), "Global":(error_globalcorr, global_confidence_dict, "solid", global_col), "Regional":(error_corr, regional_cofidence_dict, "solid", regional_col), "Random":(np.array(error_random).T, random_confidence_dict, "dashdot", "black")}

        plots.confidence_plot(analogue_vector, error_conf_dict, settings, climatol, persist_err)


    # -----------------------
    # Persistence
    error_persist = np.repeat(np.array([persist_err]), len_analogues)
    if settings["extremes_percentile"] !=0:
        error_persist = error_persist * np.nan

    ### Printing the amount of time it took
    time_end = time.perf_counter()
    print(f"    timer = {np.round(time_end - time_start, 1)} seconds")
    print('')

        # -----------------------
    # Max Skill
    if len(np.shape(soi_output)) > 1 and not ignore_baselines:
        error_maxskill = np.zeros((len_analogues, soi_input.shape[0])).T * np.nan
        with Pool(n_processes) as pool:
            no_weights = np.ones(np.shape(soi_output[:,:,:,np.newaxis])[1:])
            soi_iterable_instance = soi_iterable(n_analogues,
                                                    soi_output[:,:,:,np.newaxis],
                                                    soi_output,
                                                    analog_output[:,:,:,np.newaxis],
                                                    analog_output,
                                                    no_weights)
            if (settings["median"] or settings["percentiles"]!=None):
                error_maxskill[:, :] = run_complex_operations(metrics.super_classification_operation,
                                                                soi_iterable_instance,
                                                                pool,
                                                                chunksize=soi_input.shape[0]//n_processes,)
            elif settings["error_calc"] == "classify":
                error_maxskill[:, :] = run_complex_operations(metrics.classification_operation,
                                                                soi_iterable_instance,
                                                                pool,
                                                                chunksize=soi_input.shape[0]//n_processes,)
            elif settings["error_calc"] == "map":
                    #number of analogs x time x lat x lon
                    error_maxskill = run_complex_operations(metrics.map_operation,
                                                                soi_iterable_instance,
                                                                pool,
                                                                chunksize=soi_input.shape[0]//n_processes,)
            elif settings["error_calc"] == "field":
                error_maxskill[:, :] = run_complex_operations(metrics.field_operation,
                                                                soi_iterable_instance,
                                                                pool,
                                                                chunksize=soi_input.shape[0]//n_processes,)
            else:
                error_maxskill[:, :] = run_complex_operations(metrics.mse_operation,
                                                                soi_iterable_instance,
                                                                pool,
                                                                chunksize=soi_input.shape[0]//n_processes,)
    else:
        error_maxskill = np.zeros((len_analogues, soi_input.shape[0])).T



    # -----------------------



    if settings["error_calc"] != "map":
    ### Transpose all the error objects with a num_analogs dimension
    # Dims should be num_analogs x num_samples
        error_network = error_network.T
        error_corr = error_corr.T
        prediction_crps = prediction_crps.T
        global_crps = global_crps.T
        regional_crps = regional_crps.T
        NH_crps = NH_crps.T
        if ignore_baselines:
            error_customcorr = np.ones((len_analogues, soi_input.shape[0])) * np.nan
            #error_globalcorr = np.ones((len_analogues, soi_input.shape[0])) * np.nan
            error_globalcorr = error_globalcorr.T
            error_maxskill = np.zeros((len_analogues, soi_input.shape[0])) * np.nan
            error_random = np.zeros((len_analogues, soi_input.shape[0])) * np.nan
        else:
            error_customcorr = error_customcorr.T
            error_globalcorr = error_globalcorr.T
            error_maxskill = error_maxskill.T


    # -------------------------------------------
    # SUMMARY STATISTICS
        for i_analogue_loop, k_analogues in enumerate(analogue_vector):
            print("n_analogues = " + str(k_analogues))
            print('    network : ' + str(metrics.eval_function(error_network[i_analogue_loop, :]).round(3)))
            print(' targetcorr : ' + str(metrics.eval_function(error_corr[i_analogue_loop, :]).round(3)))
            print(' customcorr : ' + str(metrics.eval_function(error_customcorr[i_analogue_loop, :]).round(3)))
            print(' globalcorr : ' + str(metrics.eval_function(error_globalcorr[i_analogue_loop, :]).round(3)))
            print('     random : ' + str(metrics.eval_function(error_random[i_analogue_loop, :]).round(3)))
            print('      climo : ' + str(metrics.eval_function(error_climo[:]).round(3)))
            print('     custom : ' + str(metrics.eval_function(error_custom[:]).round(3)))
            print('    persist : ' + str(metrics.eval_function(error_persist[:]).round(3)))
            print('    max skill error : ' + str(metrics.eval_function(error_maxskill[i_analogue_loop,:]).round(3)))
            print('')

        # SAVE TO DICTIONARY
        metrics_dict = {
            "analogue_vector": analogue_vector,
            "error_random": error_random,
            "error_climo": error_climo,
            "error_persist": error_persist,
            "error_globalcorr": error_globalcorr,
            "error_corr": error_corr,
            "error_customcorr": error_customcorr,
            "error_network": error_network,
            "error_custom": error_custom,
            "error_maxskill": error_maxskill,
        }
        if type(error_climo_crps) != type(None):
            crps_dict = { "analogue_vector": analogue_vector,
                "error_random": np.ones_like(error_random)*np.nan,
                "error_climo": np.mean(error_climo_crps),
                "error_persist": np.ones_like(error_persist)*np.nan,
                "error_globalcorr": global_crps,
                "error_corr": regional_crps,
                "error_customcorr": NH_crps,
                "error_network":  prediction_crps,
                "error_custom": np.ones_like(error_custom)*np.nan,
                "error_maxskill": np.ones_like(error_maxskill)*np.nan,}
        else:
            crps_dict = {}

        # MAKE MAE-SKILL PLOT
        plt.figure(figsize=(8, 4))
        plots.summarize_skill_score(metrics_dict, settings)
        # plt.text(0.0, .99, ' ' + settings["savename_prefix"] + '\n smooth_time: [' + str(settings["smooth_len_input"])
        #         + ', ' + str(settings["smooth_len_output"]) + '], leadtime: ' + str(settings["lead_time"]),
        #         fontsize=6, color="gray", va="top", ha="left", fontfamily="monospace", transform=plt.gca().transAxes)
        plt.tight_layout()
        if save_figure:
            plt.savefig(dir_settings["figure_diag_directory"] + settings["savename_prefix"] +
                        '_' + fig_savename + '.png', dpi=dpiFig, bbox_inches='tight')
            plt.close()
        if show_figure:
            plt.show()
        else:
            plt.close()

        # MAKE CRPS-SKILL PLOT
        plt.figure(figsize=(8, 4))
        plots.summarize_skill_score(crps_dict, settings, 1)
        # plt.text(0.0, .99, ' ' + settings["savename_prefix"] + '\n smooth_time: [' + str(settings["smooth_len_input"])
        #         + ', ' + str(settings["smooth_len_output"]) + '], leadtime: ' + str(settings["lead_time"]),
        #         fontsize=6, color="gray", va="top", ha="left", fontfamily="monospace", transform=plt.gca().transAxes)
        plt.tight_layout()
        if save_figure:
            plt.savefig(dir_settings["figure_diag_directory"] + settings["savename_prefix"] +
                        '_' + "CRPS" + '.png', dpi=dpiFig, bbox_inches='tight')
            plt.close()
        if show_figure:
            plt.show()
        else:
            plt.close()

        return metrics_dict, crps_dict
    else:
        metrics_dict = {
            "analogue_vector": analogue_vector,
            "error_random": np.ones_like(error_globalcorr.T)*np.nan,
            "error_climo": np.tile(np.mean(error_climo,axis=(-2,-1))[:, np.newaxis], (1, np.shape(error_globalcorr)[1])).T,
            "error_persist": error_persist,
            "error_globalcorr": np.mean(error_globalcorr,axis=(-2,-1)).T,
            "error_corr": np.mean(error_corr,axis=(-2,-1)).T,
            "error_customcorr": error_customcorr.T,
            "error_network": np.mean(error_network,axis=(-2,-1)).T,
            "error_custom": np.ones_like(error_globalcorr.T)*np.nan,
            "error_maxskill": error_maxskill.T,
        }

        if type(error_climo_crps) != type(None):
            crps_dict = { "analogue_vector": analogue_vector,
                "error_random": np.ones_like(error_random)*np.nan,
                "error_climo": np.tile(np.mean(error_climo_crps,axis=(-2,-1))[:, np.newaxis], (1, np.shape(error_globalcorr)[1])).T,
                "error_persist": np.ones_like(error_persist)*np.nan,
                "error_globalcorr": np.mean(global_crps, axis=(-2,-1)).T,
                "error_corr": np.mean(regional_crps, axis=(-2,-1)).T,
                "error_customcorr": NH_crps.T,
                "error_network":  np.mean(prediction_crps, axis=(-2,-1)).T,
                "error_custom": np.ones_like(error_custom)*np.nan,
                "error_maxskill": np.ones_like(error_maxskill)*np.nan,}
        else:
            crps_dict = {}
        error_network = np.array(error_network)
        error_corr = np.array(error_corr)
        error_customcorr = np.array(error_customcorr)
        error_globalcorr = np.array(error_globalcorr)
        error_climo = np.array(error_climo)
        error_random = np.mean(error_random,axis=0)
        skill_diff_net = np.mean(error_network,axis=0)[0]
        skill_diff_net = skill_diff_net-error_climo
        skill_diff_corr = np.mean(error_corr,axis=0)[0]
        skill_diff_corr = skill_diff_corr-error_climo
        skill_diff = skill_diff_net-skill_diff_corr
        #map_out = expand_maps(lat, lon, skill_diff, settings)
       #map_skill_plot(settings, map_out, lat, lon, 10, "skill difference between net and regional")
        for analog_idx in range(1, len(analogue_vector)):
            net_err_i = np.mean(error_network,axis=0)[analog_idx]
            for nmid, error_type in enumerate([error_climo, error_corr,error_globalcorr]):
                nms = ["Climatology: ", "Regional: ", "Global: "]
                error_i = np.mean(error_type,axis=0)[analog_idx]
                skill_err = 1 - (net_err_i/error_i)
                map_out = expand_maps(lat, lon, skill_err, settings)
                map_skill_plot(settings, map_out, lat, lon, settings["analogue_vec"][analog_idx], str(nms[nmid]))
        print("#print maps of errors!")
                # MAKE MAE-SKILL PLOT
        plt.figure(figsize=(8, 4))
        plots.summarize_skill_score(metrics_dict, settings)
        # plt.text(0.0, .99, ' ' + settings["savename_prefix"] + '\n smooth_time: [' + str(settings["smooth_len_input"])
        #         + ', ' + str(settings["smooth_len_output"]) + '], leadtime: ' + str(settings["lead_time"]),
        #         fontsize=6, color="gray", va="top", ha="left", fontfamily="monospace", transform=plt.gca().transAxes)
        plt.tight_layout()
        if save_figure:
            plt.savefig(dir_settings["figure_diag_directory"] + settings["savename_prefix"] +
                        '_' + fig_savename + '.png', dpi=dpiFig, bbox_inches='tight')
            plt.close()
        if show_figure:
            plt.show()
        else:
            plt.close()
                # MAKE CRPS-SKILL PLOT
        plt.figure(figsize=(8, 4))
        plots.summarize_skill_score(crps_dict, settings, 1)
        plt.tight_layout()
        if save_figure:
            plt.savefig(dir_settings["figure_diag_directory"] + settings["savename_prefix"] +
                        '_' + "CRPS" + '.png', dpi=dpiFig, bbox_inches='tight')
            plt.close()
        if show_figure:
            plt.show()
        else:
            plt.close()

        return metrics_dict, crps_dict

def retrieve_mask(model, settings, shape, setmaskedas=0.0):
    weighted_mask = model.get_layer('mask_model').get_layer("weights_layer").bias.numpy()
    if shape is not None:
        weighted_mask = weighted_mask.reshape(shape)
    geomask = build_data.mask_in_land_ocean(None, settings, maskout=settings["maskout_landocean_input"]).squeeze()

    weighted_mask = scale_mask(weighted_mask, geomask, setmaskedas=setmaskedas)

    return weighted_mask #returns a weighted mask that has been filtered based on the input land/ocean mask as well

def scale_mask(weighted_mask, landmask, setmaskedas=0.0):
    # Set masked values to zero
    weighted_mask[landmask == 0] = 0.0
    num_weights = np.sum(landmask) * weighted_mask.shape[-1]
    # Re-scale so the mean weight is 1
    weighted_mask = weighted_mask / np.sum(weighted_mask) * num_weights
    # Set masked values to the specified value (usually 0.0 or np.nan)
    weighted_mask[landmask == 0] = setmaskedas

    return weighted_mask


def visualize_interp_model(settings, weights_train, lat, lon, sv="", clims =(0,0), ttl=""):
    #expects shape of lat x lon x channels
    num_maps = weights_train.shape[-1]
    ax = dict()
    fig = plt.figure(figsize=(7.5, 5* num_maps))

    # colorbar limits
    climits_dat = weights_train
    climits = (climits_dat.min(), climits_dat.max())
    if clims != (0,0): 
        climits = clims

    print(climits)

    reg_code = settings["target_region_name"]
    if reg_code == "n_atlantic_ext":
        reg_name = "the North Atlantic"
    elif reg_code == "california":
        reg_name = "Southern California"
    elif reg_code == "midwest":
        reg_name = "the Midwestern U.S."

    # plot the weighted mask
    if len(weights_train.shape) == 4:

        weights_train = weights_train.mean(axis=0)
        for imap in range(num_maps):
            if imap == 0:
                predictor = settings["feature_var"]
                if predictor == "U":
                    lab = "U250"
                if predictor == "TREFHT":
                    lab = "Temperature"
                if predictor == "PRECT":
                    lab = "Precipitation"
            else:
                predictor = settings["extra_channel"][imap-1]
                if predictor == "U":
                    lab = "U250"
                if predictor == "TREFHT":
                    lab = "Temperature"
                if predictor == "PRECT":
                    lab = "Precipitation"
            if type(ttl) == type(""):
                ttl_text = lab + " Mask for " + reg_name
            else:
                ttl_text = str(imap) + " Branch Weight: " + str(ttl[imap])
            ax, _ = plots.plot_state_masks(fig, settings, weights_train_ind[:, :, imap], lat=lat, lon=lon, central_longitude=215., climits = climits, title_text=ttl_text,subplot=(1, num_maps, imap + 1), )
    else:
        for imap in range(num_maps):
            if imap == 0:
                predictor = settings["feature_var"]
                if predictor == "U":
                    lab = "U250"
                if predictor == "TREFHT":
                    lab = "Temperature"
                if predictor == "PRECT":
                    lab = "Precipitation"
            else:
                predictor = settings["extra_channel"][imap-1]
                if predictor == "U":
                    lab = "U250"
                if predictor == "TREFHT":
                    lab = "Temperature"
                if predictor == "PRECT":
                    lab = "Precipitation"
            if type(ttl) == type(""):
                ttl_text = ttl_text = lab + " Mask for " + reg_name
            else:
                ttl_text = str(imap) + " Branch Weight: " + str(ttl[imap])
            ax, _ = plots.plot_interp_masks(fig, settings, weights_train[:, :, imap], lat=lat, lon=lon, 
                                            central_longitude=215., climits=climits, title_text=ttl_text, 
                                            subplot=(num_maps, 1, imap + 1), use_text=False)

 # save the mask
        print(dir_settings["figure_directory"] + settings["savename_prefix"] +
                    '_averaged_masks.png')
        print('./figures/')
        plt.tight_layout()
        if sv == "":
            plt.savefig(dir_settings["figure_directory"] + settings["savename_prefix"] +
                    '_averaged_masks.png', dpi=dpiFig, bbox_inches='tight')
        else:
            plt.savefig(dir_settings["figure_directory"] + settings["savename_prefix"] + sv +
                    '_averaged_masks.png', dpi=dpiFig, bbox_inches='tight')
        plt.close()
def map_skill_plot(settings, weights_train, lat, lon, analog_vector, name, extent_limit = 1, crps = 0):
    #expects shape of lat x lon x channels
    num_maps = 1
    if settings["soi_test_members"][0] == "ERA5":
        prefix = "ERA5"
    else:
        prefix = "CESM"
    ax = dict()
    fig = plt.figure(figsize=(7.5 * num_maps, 5))
    sv_addition = ""
    # colorbar limits
    climits_dat = weights_train
    if crps:
        ttl = prefix + " CRPS Skill for " + name + " " + str(analog_vector) + ' analogs'
        sv_addition = "_CRPS"
    else:
        ttl = prefix + " MAE Skill Relative to " + name + " " + str(analog_vector) + ' analogs'
    climits = (-np.max([np.abs(np.quantile(climits_dat,.10)),np.quantile(climits_dat,.90)]), np.max([np.abs(np.quantile(climits_dat,.10)),np.quantile(climits_dat,.90)]))
    # plot the weighted mask
    for imap in range(num_maps):
        ax, _ = plots.JBL_maps_plot(fig, settings, weights_train[:, :], lat=lat, lon=lon, central_longitude=215., climits = climits, title_text=ttl,subplot=(1, num_maps, imap + 1), extent_limit=extent_limit)


    # save the mask
    print(dir_settings["figure_directory"] + settings["savename_prefix"] + '_' + name + '_' + str(analog_vector) + 'analogs')
    plt.tight_layout()
    plt.savefig(dir_settings["figure_directory"] + settings["savename_prefix"] + '_' + name + '_' + str(analog_vector) + '_analogs' + sv_addition, dpi=dpiFig, bbox_inches='tight')
    plt.close()


def JBL_visualize_interp_model(settings, weights_train, lat, lon, err, mse_err):
    titles = ["target input", "target output", "analog input", "analog output"]
    num_maps = weights_train.shape[-1]
    ax = dict()
    fig = plt.figure(figsize=(15,15/1.8))
    fig.suptitle("ACC value: " + str(err) + " rmse relative to climatology: " + str(mse_err))
    ins = weights_train[:,:,(0,2)]
    outs = weights_train[:,:,(1,3)]
    # colorbar limits
    cmin_in = np.percentile(ins, 5)  # np.min(weights_train[:])
    cmax_in = np.percentile(ins, 95)
    cmin_out = np.percentile(outs, 5)  # np.min(weights_train[:])
    cmax_out = np.percentile(outs, 95)
    climits_ins = (cmin_in, cmax_in)
    climits_outs = (cmin_out, cmax_out)
    lims = [climits_ins, climits_outs, climits_ins, climits_outs]
    mps = ["test", "test", "test", "test"]
    # plot the weighted mask
    for imap in range(num_maps):
        titl = titles[imap]
        clims = lims[imap]
        mp = mps[imap]
        ax, _ = plots.JBL_plot_interp_masks(fig, settings, weights_train[:, :, imap], lat=lat, lon=lon, central_longitude=215., climits = clims, title_text=titl,subplot=(2, 2, imap + 1),cmap=mp )
    # save the mask
    plt.savefig(dir_settings["figure_directory"] + settings["savename_prefix"] +
                '_test_anlog_maps', dpi=dpiFig, bbox_inches="tight")
    plt.close()

def expand_maps(lat, lon, map, settings):
    output_map = np.zeros((np.shape(lat)[0],np.shape(lon)[0]))*np.NAN
    #find closest lat and lon
    rgn = regions.get_region_dict(settings["target_region_name"]) #pick better indices and make NaN not zeros
    rgn_low_lat = rgn['lat_range'][0]
    rgn_low_lon = rgn['lon_range'][0]
    low_lat_idx = np.argmax(lat >= rgn_low_lat) 
    low_lon_idx = np.argmax(lon >= rgn_low_lon)
    output_map[low_lat_idx:low_lat_idx + np.shape(map)[0], low_lon_idx:low_lon_idx + np.shape(map)[1]] = map
    return output_map