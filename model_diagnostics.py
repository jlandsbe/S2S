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
from scipy.stats import linregress
warnings.filterwarnings(action='ignore', message='Mean of empty slice')
warnings.filterwarnings(action='ignore', message='All-NaN slice encountered')
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
#np.warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

__author__ = "Jamin K. Rader, Elizabeth A. Barnes, and Randal J. Barnes"
__version__ = "30 March 2023"

dir_settings = base_directories.get_directories()
dpiFig = 300


def visualize_metrics(settings, model, soi_input, soi_output, analog_input, analog_output, 
                      lat, lon, mask, persist_err=0, n_testing_analogs=1_000,
                      analogue_vector=None, fig_savename="",
                      soi_train_output=None, my_masks = None):
    if analogue_vector is None:
        analogue_vector = [1, 2, 5, 10, 15, 20, 25, 30]

    rng_eval = np.random.default_rng(seed=settings["rng_seed"]+5)

    # define analog_set size, either all data, or specified amount in experiments.py
    n_testing_analogs = np.min([analog_input.shape[0], n_testing_analogs])
    # get random analogs
    i_analog = rng_eval.choice(np.arange(0, analog_input.shape[0]), n_testing_analogs, replace=False)

    # assess model performance and compare to baselines
    metrics_dict = assess_metrics(settings, model,
                                  soi_input[:, :, :, :],
                                  soi_output[:],
                                  analog_input[i_analog, :, :, :],
                                  analog_output[i_analog],
                                  lat, lon,
                                  mask, persist_err,
                                  soi_train_output=soi_train_output,
                                  analogue_vector=analogue_vector,
                                  fig_savename=fig_savename, my_masks = my_masks)

    return metrics_dict


def run_complex_operations(operation, inputs, pool, chunksize):
    return pool.map(operation, inputs, chunksize=chunksize)


def soi_iterable(n_analogs, soi_input, soi_output, analog_input, analog_output, mask, uncertainties = 0):
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


def assess_metrics(settings, model, soi_input, soi_output, analog_input,
                   analog_output, lat, lon,
                   mask, persist_err=0,
                   soi_train_output=None,
                   analogue_vector=[1, 2, 5, 10, 15, 20, 25, 30],
                   show_figure=False, save_figure=True, fig_savename="", my_masks=None):

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

    # Which analogs are we going to go through?
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
                    soi_iterable_instance = soi_iterable(n_analogues,
                                                            soi_input,
                                                            soi_output,
                                                            analog_input,
                                                            analog_output,
                                                            mask, 1)
                    if settings["error_calc"] == "super_classify":
                        error_network[:, :] = run_complex_operations(metrics.super_classification_operation,
                                                                    soi_iterable_instance,
                                                                    pool,
                                                                    chunksize=soi_input.shape[0]//n_processes,)
                    elif settings["error_calc"] == "classify":
                        error_network[:, :] = run_complex_operations(metrics.classification_operation,
                                                                    soi_iterable_instance,
                                                                    pool,
                                                                    chunksize=soi_input.shape[0]//n_processes,)
                    elif settings["error_calc"] == "map":
                        #number of analogs x lat x lon
                        x  = run_complex_operations(metrics.map_operation,
                                                                    soi_iterable_instance,
                                                                    pool,
                                                                    chunksize=soi_input.shape[0]//n_processes,)
                        error_network = x[0]
                        analog_match_error = x[1]
                        prediction_spread = x[2]
                    elif settings["error_calc"] == "field":
                        x = np.array(run_complex_operations(metrics.field_operation,
                                                                    soi_iterable_instance,
                                                                    pool,
                                                                    chunksize=soi_input.shape[0]//n_processes,))
                        error_network[:, :] = x[:,0,:]
                        analog_match_error = x[:,1,:] 
                        prediction_spread = x[:,2,:]
                        plots.uncertainty_whiskers(analogue_vector, error_network, analog_match_error, prediction_spread, settings, bins = [0, .1, .3, 1])
                    else:
                        x = np.array(run_complex_operations(metrics.mse_operation,
                                                                    soi_iterable_instance,
                                                                    pool,
                                                                    chunksize=soi_input.shape[0]//n_processes,))
                        error_network[:, :] = x[:,0,:]
                        analog_match_error = x[:,1,:] 
                        prediction_spread = x[:,2,:]
                        plots.uncertainty_whiskers(analogue_vector, error_network, analog_match_error, prediction_spread, settings, bins = [0, .1, .3, 1])
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
    with Pool(n_processes) as pool:
        sqrt_area_weights = np.sqrt(np.cos(np.deg2rad(lat))[np.newaxis, :, np.newaxis, np.newaxis])
        soi_iterable_instance = soi_iterable(n_analogues,
                                                soi_input,
                                                soi_output,
                                                analog_input,
                                                analog_output,
                                                sqrt_area_weights)
        if settings["error_calc"] == "super_classify":
            error_globalcorr[:, :] = run_complex_operations(metrics.super_classification_operation,
                                                            soi_iterable_instance,
                                                            pool,
                                                            chunksize=soi_input.shape[0]//n_processes,)
        elif settings["error_calc"] == "classify":
            error_globalcorr[:, :] = run_complex_operations(metrics.classification_operation,
                                                            soi_iterable_instance,
                                                            pool,
                                                            chunksize=soi_input.shape[0]//n_processes,)
        elif settings["error_calc"] == "map":
                #number of analogs x time x lat x lon
                error_globalcorr = run_complex_operations(metrics.map_operation,
                                                            soi_iterable_instance,
                                                            pool,
                                                            chunksize=soi_input.shape[0]//n_processes,)
        elif settings["error_calc"] == "field":
            error_globalcorr[:, :] = run_complex_operations(metrics.field_operation,
                                                            soi_iterable_instance,
                                                            pool,
                                                            chunksize=soi_input.shape[0]//n_processes,)
        else:
            error_globalcorr[:, :] = run_complex_operations(metrics.mse_operation,
                                                            soi_iterable_instance,
                                                            pool,
                                                            chunksize=soi_input.shape[0]//n_processes,)
        print("finished global error")
    # -----------------------
    # Simple TARGET REGION correlation baseline
    with Pool(n_processes) as pool:
        soi_reg, lat_reg, lon_reg = build_data.extract_region(soi_input, regions.get_region_dict(
            settings["target_region_name"]), lat=lat, lon=lon)
        analog_reg, __, __ = build_data.extract_region(analog_input,
                                                        regions.get_region_dict(settings["target_region_name"]),
                                                        lat=lat, lon=lon)
        sqrt_area_weights = np.sqrt(np.cos(np.deg2rad(lat_reg))[np.newaxis, :, np.newaxis, np.newaxis])
        soi_iterable_instance = soi_iterable(n_analogues,
                                                soi_reg,
                                                soi_output,
                                                analog_reg,
                                                analog_output,
                                                sqrt_area_weights)
        if settings["error_calc"] == "super_classify":
            error_corr[:, :] = run_complex_operations(metrics.super_classification_operation,
                                                            soi_iterable_instance,
                                                            pool,
                                                            chunksize=soi_input.shape[0]//n_processes,)
        elif settings["error_calc"] == "map":
            #number of analogs x time x lat x lon
            error_corr = run_complex_operations(metrics.map_operation,
                                                        soi_iterable_instance,
                                                        pool,
                                                        chunksize=soi_input.shape[0]//n_processes,)
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
            error_corr[:, :] = run_complex_operations(metrics.mse_operation,
                                                            soi_iterable_instance,
                                                            pool,
                                                            chunksize=soi_input.shape[0]//n_processes,)
        print("finished target region error")
        
    # -----------------------
    # Simple CUSTOM CORRELATION REGION correlation baseline (not needed)
    if "correlation_region_name" in settings.keys() and my_masks != None:
        with Pool(n_processes) as pool:
            # soi_reg, lat_reg, lon_reg = build_data.extract_region(soi_input, regions.get_region_dict(
            #     settings["correlation_region_name"]), lat=lat, lon=lon)
            # analog_reg, __, __ = build_data.extract_region(analog_input,
            #                                             regions.get_region_dict(settings["correlation_region_name"]),
            #                                             lat=lat, lon=lon)
            # sqrt_area_weights = np.sqrt(np.cos(np.deg2rad(lat_reg))[np.newaxis, :, np.newaxis, np.newaxis])
            soi_iterable_instance = soi_iterable_masks(n_analogues,
                                                soi_input,
                                                soi_output,
                                                analog_input,
                                                analog_output,
                                                my_masks)
            if settings["error_calc"] == "super_classify":
                error_customcorr[:, :] = run_complex_operations(metrics.super_classification_operation,
                                                            soi_iterable_instance,
                                                            pool,
                                                            chunksize=soi_input.shape[0]//n_processes,)
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
                error_customcorr[:, :] = run_complex_operations(metrics.mse_operation,
                                                            soi_iterable_instance,
                                                            pool,
                                                            chunksize=soi_input.shape[0]//n_processes,)
    
    # -----------------------
    # Custom baseline (e.g. mean evolution)
    if "custom_baseline" in settings.keys():
        custom_true, custom_pred = metrics.calc_custom_baseline(settings["custom_baseline"], 
                                                                soi_output=soi_output,
                                                                soi_train_output=soi_train_output,
                                                                settings=settings)
        error_custom[:] = metrics.get_analog_errors(custom_true, custom_pred)

    # -----------------------
    # Random baseline
    for idx_analog, n_analog in enumerate(n_analogues):
        i_analogue = rng.choice(np.arange(0, analog_output.shape[0]),
                                size=(n_analog, soi_output.shape[0]), replace=True)
        if settings["error_calc"] == "super_classify":
            error_random[idx_analog, :] = metrics.get_analog_errors(soi_output,
                                                    np.median(analog_output[i_analogue], axis=0), settings["error_calc"])
        elif settings["error_calc"] == "map":
            np.append(error_random,metrics.get_analog_errors(soi_output,
                                                    np.median(analog_output[i_analogue], axis=0), settings["error_calc"]))
        else: 
            error_random[idx_analog, :] = metrics.get_analog_errors(soi_output,
                                                    np.mean(analog_output[i_analogue], axis=0), settings["error_calc"])

    # -----------------------
    # Climatology
    if settings["error_calc"] == "super_classify":
        error_climo[:] = metrics.get_analog_errors(soi_output, np.median(analog_output), settings["error_calc"]).T
    elif settings["error_calc"] == "map":
        error_climo = metrics.get_analog_errors(soi_output, np.mean(analog_output, axis=0), settings["error_calc"])
    else:
        error_climo[:] = metrics.get_analog_errors(soi_output, np.mean(analog_output, axis=0), settings["error_calc"]).T

    # -----------------------
    # Persistence
    #persist_true, persist_pred = metrics.calc_persistence_baseline(soi_output, settings)
    error_persist = np.repeat(np.array([persist_err]), len_analogues)

    ### Printing the amount of time it took
    time_end = time.perf_counter()
    print(f"    timer = {np.round(time_end - time_start, 1)} seconds")
    print('')
    if settings["error_calc"] != "map":
    ### Transpose all the error objects with a num_analogs dimension
    # Dims should be num_analogs x num_samples
        error_network = error_network.T
        error_corr = error_corr.T
        error_customcorr = error_customcorr.T
        error_globalcorr = error_globalcorr.T

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
        }

        # MAKE SUMMARY-SKILL PLOT
        plt.figure(figsize=(8, 4))
        plots.summarize_skill_score(metrics_dict, settings["error_calc"])
        plt.text(0.0, .99, ' ' + settings["savename_prefix"] + '\n smooth_time: [' + str(settings["smooth_len_input"])
                + ', ' + str(settings["smooth_len_output"]) + '], leadtime: ' + str(settings["lead_time"]),
                fontsize=6, color="gray", va="top", ha="left", fontfamily="monospace", transform=plt.gca().transAxes)
        plt.tight_layout()
        if save_figure:
            plt.savefig(dir_settings["figure_diag_directory"] + settings["savename_prefix"] +
                        '_' + fig_savename + '.png', dpi=dpiFig, bbox_inches='tight')
            plt.close()
        if show_figure:
            plt.show()
        else:
            plt.close()

        return metrics_dict
    else:
        #all the errors I get will be of the form analogs x lat x lon,
        #so I want to input the analog vector for the title and the numpy arrays as weights, number of maps will always be 1, just
        #do a four loop where I save with the number of analogs in the name of the thing
        #also I will want to do the numpy subtraction of climatology to get a skill before passing it to my plot function
        #climo_tiled = np.broadcast_to(error_climo, np.shape(error_network))
        #error_persist = np.broadcast_to(error_persist, np.shape(error_network))
        id = 0
        nmid = 0
        error_network = np.array(error_network)
        error_corr = np.array(error_corr)
        error_customcorr = np.array(error_customcorr)
        error_globalcorr = np.array(error_globalcorr)
        error_climo = np.mean(error_climo,axis=0)
        error_random = np.mean(error_random,axis=0)
        skill_diff_net = np.mean(error_network,axis=0)[0]
        skill_diff_net = skill_diff_net-error_climo
        skill_diff_corr = np.mean(error_corr,axis=0)[0]
        skill_diff_corr = skill_diff_corr-error_climo
        skill_diff = skill_diff_net-skill_diff_corr
        #map_out = expand_maps(lat, lon, skill_diff, settings)
       #map_skill_plot(settings, map_out, lat, lon, 10, "skill difference between net and regional")
        for error_type in [error_network, error_corr]:
            nms = ["network", "regional"]
            for an_number in analogue_vector:
                print(np.shape(error_type))
                error_i = np.mean(error_type,axis=0)[id]
                skill_err = 1 - (error_i/error_climo)
                skill_err = error_i
                map_out = expand_maps(lat, lon, skill_err, settings)
                map_out = error_i
                map_skill_plot(settings, map_out, lat, lon, an_number, str(nms[nmid]))
                id = id+1
            nmid = nmid+1
            id = 0
        print("#print maps of errors!")

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

def video_syn_data(settings, weights_train_list, lat, lon, sv=""):
    #expects shape of lat x lon x channels
    num_maps = [weights_train_list][0].shape[-1]
    ax = dict()
    fig = plt.figure(figsize=(7.5 * num_maps, 5))

    # colorbar limits
    #climits_dat = np.squeeze(weights_train_list[0,:,:,:])
    climits = (np.nanmin(weights_train_list),np.nanmax(weights_train_list))
    # plot the weighted mask
    def update(frame):
        plt.clf()  # Clear the previous plot
        weights_train = weights_train_list[frame]
        artists = []
        if len(weights_train.shape) == 4:
            for imap in range(num_maps):
                ax, artist = plots.plot_state_masks(fig, settings, weights_train_ind[:, :, imap], lat=lat, lon=lon, central_longitude=215., climits = climits, title_text="Mask for Channel " + str(imap),subplot=(1, num_maps, imap + 1), )
                artists.append(artist)
        else:
            for imap in range(num_maps):
                ax, artist = plots.plot_interp_masks(fig, settings, weights_train[:, :, imap], lat=lat, lon=lon, central_longitude=215., climits = climits, title_text="Mask for Channel " + str(imap),subplot=(1, num_maps, imap + 1), )
                artists.append(artist)
        return artists
    # save the mask
    ani = animation.FuncAnimation(fig, update, frames=len(weights_train_list), interval=50, blit=False)
    output_file = dir_settings["figure_directory"] + settings["savename_prefix"] + sv + "_synthetic_data.gif"
    ani.save(output_file, writer='imagemagick')
    plt.tight_layout()

def visualize_interp_model(settings, weights_train, lat, lon, sv="", clims =(0,0), ttl=""):
    #expects shape of lat x lon x channels
    num_maps = weights_train.shape[-1]
    ax = dict()
    fig = plt.figure(figsize=(7.5 * num_maps, 5))

    # colorbar limits
    climits_dat = weights_train
    climits = (climits_dat.min(), climits_dat.max())
    if clims != (0,0): 
        climits = clims
    #climits = (np.percentile(climits_dat, 96),np.percentile(climits_dat, 99))
    # plot the weighted mask
    print("in the thing")
    print(climits)
    if len(weights_train.shape) == 4:

        weights_train = weights_train.mean(axis=0)
        for imap in range(num_maps):
            if type(ttl) == type(""):
                ttl_text = "Mask for Channel " + str(imap)
            else:
                ttl_text = str(imap) + " Branch Weight: " + str(ttl[imap])
            ax, _ = plots.plot_state_masks(fig, settings, weights_train_ind[:, :, imap], lat=lat, lon=lon, central_longitude=215., climits = climits, title_text=ttl_text,subplot=(1, num_maps, imap + 1), )
    else:
        for imap in range(num_maps):
            if type(ttl) == type(""):
                ttl_text = "Mask for Channel " + str(imap)
            else:
                ttl_text = str(imap) + " Branch Weight: " + str(ttl[imap])
            ax, _ = plots.plot_interp_masks(fig, settings, weights_train[:, :, imap], lat=lat, lon=lon, central_longitude=215., climits = climits, title_text=ttl_text,subplot=(1, num_maps, imap + 1), )
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
def map_skill_plot(settings, weights_train, lat, lon, analog_vector, name):
    #expects shape of lat x lon x channels
    num_maps = 1
    ax = dict()
    fig = plt.figure(figsize=(7.5 * num_maps, 5))

    # colorbar limits
    climits_dat = weights_train

    climits = (-np.max([np.abs(np.quantile(climits_dat,.10)),np.quantile(climits_dat,.90)]), np.max([np.abs(np.quantile(climits_dat,.10)),np.quantile(climits_dat,.90)]))
    # plot the weighted mask
    for imap in range(num_maps):
        ax, _ = plots.JBL_maps_plot(fig, settings, weights_train[:, :], lat=lat, lon=lon, central_longitude=215., climits = climits, title_text="MSE Skill for " + name + " " + str(analog_vector) + ' analogs',subplot=(1, num_maps, imap + 1))

    # save the mask
    print(dir_settings["figure_directory"] + settings["savename_prefix"] + '_' + name + '_' + str(analog_vector) + 'analogs')
    plt.tight_layout()
    plt.savefig(dir_settings["figure_directory"] + settings["savename_prefix"] + '_' + name + '_' + str(analog_vector) + '_analogs', dpi=dpiFig, bbox_inches='tight')
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