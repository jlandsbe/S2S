"""Metrics.

Functions
---------

"""

import numpy as np
import tensorflow as tf
import build_data
import tensorflow_probability as tfp

__author__ = "Jamin K. Rader, Elizabeth A. Barnes, and Randal J. Barnes"
__version__ = "30 March 2023"

def compute_best_analogs(inputs, n_analogs):
    #the shape here will be lat x lon x number of channels for the inputs/masks
    mimse = (np.mean((inputs["soi_input_sample"]*inputs["mask"] - inputs["analog_input"]*inputs["mask"]) ** 2,
                     axis=(1, 2, 3)))
    i_analogs = np.argsort(mimse, axis=0)[:n_analogs]
    return i_analogs  #this returns the best analogs' indices based on mimse

#make change
def mse_operation(inputs):
    assert type(inputs["n_analogs"]) is not int # should be a list-type
    i_analogs = compute_best_analogs(inputs, inputs["max_analogs"])
    results = []
    if inputs["uncertainties"]:
        input_diff = []
        output_spread = []
        for n_analogs in inputs["n_analogs"]:
            results.append(get_analog_errors(inputs["soi_output_sample"], np.mean(inputs["analog_output"][i_analogs[:n_analogs]]), "mse"))
            input_diff.append((np.mean((inputs["soi_input_sample"] - inputs["analog_input"][i_analogs[:n_analogs]])**2))**.5)
            output_spread.append(np.mean(np.var(inputs["analog_output"][i_analogs[:n_analogs]],axis=0)))
        return np.stack(results, axis=0), np.stack(input_diff, axis=0), np.stack(output_spread, axis=0)
    else:
        for n_analogs in inputs["n_analogs"]:
            results.append(get_analog_errors(inputs["soi_output_sample"], np.mean(inputs["analog_output"][i_analogs[:n_analogs]]), "mse"))
        return np.stack(results, axis=0)

def field_operation(inputs):
    assert type(inputs["n_analogs"]) is not int # should be a list-type
    i_analogs = compute_best_analogs(inputs, inputs["max_analogs"])
    results = []
    if inputs["uncertainties"]:
        input_diff = []
        output_spread = []
        for n_analogs in inputs["n_analogs"]:
            results.append(get_analog_errors(inputs["soi_output_sample"], np.mean(inputs["analog_output"][i_analogs[:n_analogs]], axis=0), "field"))
            input_diff.append((np.mean((inputs["soi_input_sample"] - inputs["analog_input"][i_analogs[:n_analogs]])**2))**.5)
            output_spread.append(np.mean(np.var(inputs["analog_output"][i_analogs[:n_analogs]],axis=0)))
        return [np.stack(results, axis=0), np.stack(input_diff, axis=0), np.stack(output_spread, axis=0)]
    else:
        for n_analogs in inputs["n_analogs"]:
            results.append(get_analog_errors(inputs["soi_output_sample"], np.mean(inputs["analog_output"][i_analogs[:n_analogs]], axis=0), "field"))
        return np.stack(results, axis=0)

def map_operation(inputs):
    assert type(inputs["n_analogs"]) is not int # should be a list-type
    i_analogs = compute_best_analogs(inputs, inputs["max_analogs"])
    results = []
    if inputs["uncertainties"]:
        input_diff = []
        output_spread = []
        for n_analogs in inputs["n_analogs"]:
            results.append(get_analog_errors(inputs["soi_output_sample"], np.mean(inputs["analog_output"][i_analogs[:n_analogs]], axis=0), "map"))
            input_diff.append((np.mean((inputs["soi_input_sample"] - inputs["analog_input"][i_analogs[:n_analogs]])**2))**.5)
            output_spread.append(np.mean(np.var(inputs["analog_output"][i_analogs[:n_analogs]],axis=0)))
        return [np.stack(results, axis=0), np.stack(input_diff, axis=0), np.stack(output_spread, axis=0)]
    else:
        for n_analogs in inputs["n_analogs"]:
            results.append(get_analog_errors(inputs["soi_output_sample"], np.mean(inputs["analog_output"][i_analogs[:n_analogs]], axis=0), "map"))
        return np.stack(results, axis=0)

def map_operation2(inputs):
    assert type(inputs["n_analogs"]) is not int # should be a list-type
    i_analogs = inputs["i_analogs"]
    results = []
    for n_analogs in inputs["n_analogs"]:
        results.append(get_analog_errors(inputs["soi_output_sample"], np.mean(inputs["analog_output"][i_analogs[:n_analogs]], axis=0), "map"))
    return np.stack(results, axis=0) #analog x lat x lon

def test_operation(inputs):
    assert type(inputs["n_analogs"]) is not int # should be a list-type
    i_analogs = compute_best_analogs(inputs, inputs["max_analogs"])
    results = []
    input_maps = np.mean(inputs["analog_input"][i_analogs[:10]], axis=0)
    output_maps = np.mean(inputs["analog_output"][i_analogs[:10]], axis=0)
    for n_analogs in inputs["n_analogs"]:
        results.append(get_analog_errors(inputs["soi_output_sample"], np.mean(inputs["analog_output"][i_analogs[:n_analogs]], axis=0), "field"))
    for n_analogs in inputs["n_analogs"]: 
        mse = get_analog_errors(inputs["soi_output_sample"], np.mean(inputs["analog_output"][i_analogs[:n_analogs]], axis=0), "mse")
    clima = get_analog_errors(inputs["soi_output_sample"], np.mean(inputs["analog_output"], axis=0), "mse")
    mse = 1-(mse/clima)
    return input_maps, output_maps, inputs["soi_input_sample"], inputs["soi_output_sample"], 1-results[0], mse

def classification_operation(inputs):
    i_analogs = compute_best_analogs(inputs, inputs["max_analogs"])
    results = []
    for n_analogs in inputs["n_analogs"]:
        results.append(get_analog_errors(inputs["soi_output_sample"], np.mean(inputs["analog_output"][i_analogs[:n_analogs]]), "classify")*1.0)
    return np.stack(results, axis=0)

def super_classification_operation(inputs):
    i_analogs = compute_best_analogs(inputs, inputs["max_analogs"])
    results = []
    for n_analogs in inputs["n_analogs"]:
        results.append(get_analog_errors(inputs["soi_output_sample"], np.round(np.mean(inputs["analog_output"][i_analogs[:n_analogs]], axis=0)), "mae"))
    return np.stack(results, axis=0)

#do we want this to be weighted?
def mse(analog, truth):
    difference = tf.keras.layers.subtract([analog, truth])
    difference_squared = tf.keras.backend.square(difference)
    return tf.math.reduce_mean(difference_squared, axis=-1)

def anomaly_correlation(analog, truth):
    # Calculate anomaly correlation using Pearson correlation coefficient
    anomaly_corr = tfp.stats.correlation(analog, truth)
    return 1.0 - anomaly_corr  

def pred_range(y_true, y_pred):
    return tf.math.reduce_max(y_pred) - tf.math.reduce_min(y_pred)


def get_persist_errors(soi, analog_prediction, type="mse"):
    if type=="classify":
        return get_analog_errors_pos_neg(soi, analog_prediction)
    if type=="mse":
        diff = (soi - analog_prediction)**2
        if len(np.shape(diff))>2:
            diff = np.mean(diff, axis = (-1,-2))#think I should change this to axis = (1,2) to account for other channel in last indice
        return diff**.5
    if type=="field":
        numerator = np.sum((soi * analog_prediction), axis=(-1,-2))
        denominator_x = np.sum((soi*soi), axis=(-1,-2))
        denominator_y = np.sum((analog_prediction*analog_prediction),  axis=(-1,-2))
        weighted_corr = numerator / (np.sqrt(denominator_x * denominator_y))
        return 1-weighted_corr
    if type=="super_classify":
        return np.array([.5])
    if type == "map":
        raise Exception("Sorry, this error calculation type is not implemented.")
    else:
        raise Exception("Sorry, this error calculation type is not implemented.")

def get_analog_errors(soi, analog_prediction, type="mse"):
    if type=="classify":
        return get_analog_errors_pos_neg(soi, analog_prediction)
    if type=="mse":
        diff = (soi - analog_prediction)**2
        if len(np.shape(diff))>1:
            #diff = np.average(diff, axis=tuple(range(1,soi.ndim)))
            diff = np.average(diff, axis=(-1,-2))#think I should change this to axis = (1,2) to account for other channel in last indice
        return diff**.5
    if type=="field":
        numerator = np.sum((soi * analog_prediction), axis=(-1,-2))
        denominator_x = np.sum((soi*soi), axis=(-1,-2))
        denominator_y = np.sum((analog_prediction*analog_prediction),  axis=(-1,-2))
        weighted_corr = numerator / (np.sqrt(denominator_x * denominator_y))
        return 1-weighted_corr
    if type=="map":
        diff = (soi - analog_prediction)**2
        return diff**.5
    if type=="mae":
        diff = np.abs(soi - analog_prediction)
        return np.average(diff, axis=(-1,-2))
    else:
        raise Exception("Sorry, this error calculation type is not implemented.")

def get_analog_errors_pos_neg(soi, analog_prediction):
    return np.logical_not(np.logical_or(((soi==analog_prediction)), (soi*analog_prediction>0)))




def get_targets(settings, soi, analog):
    if settings["model_type"] == "ann_model":
        target = analog.copy()
    else:        
        target = get_analog_errors(soi, analog, "mse") 
        if settings["output_type"] == "classification":
            target = (target >= settings["class_threshold"]).astype('float32')
    return target


def eval_function(x):
    return np.nanmean(x, axis=-1)

### Persistence Baseline

def calc_persistence_baseline(soi_output, settings, cont_arr_key = "soi_test_members"):
        # to account for the fact the data is non-continuous
        num_segments = len(settings[cont_arr_key])
        split_outputs = np.split(soi_output, num_segments)

        # need to account for the smoothing length of the data
        num_smoothing = np.abs(settings["smooth_len_output"])

        persistence_truth_list = []
        persistence_prediction_list = []

        for cont_arr in split_outputs:
            persistence_truth_list.append(cont_arr[num_smoothing:]) # all but first of each continuous arr
            persistence_prediction_list.append(cont_arr[:-num_smoothing]) # all but last of each cont arr

        persistence_truth = np.concatenate(persistence_truth_list)
        persistence_prediction = np.concatenate(persistence_prediction_list)
        
        # returns the truths AND predictions
        return persistence_truth, persistence_prediction

### Custom Baselines
def calc_custom_baseline(name, soi_output=None, soi_train_output=None, settings=None):

    # The average evolution. Given the target in year 0, what is the expected targer value in year 1?
    if name == "avg_evolution":
        soi_test_output = soi_output
        soi_train_output = soi_train_output

        s_in_train, s_out_train = calc_persistence_baseline(soi_train_output, settings, 
                                                            cont_arr_key="soi_train_members")
        s_in_test, s_out_test = calc_persistence_baseline(soi_test_output, settings)

        samples_per_bin = 50

        # Bin the input data
        percentile_bins = np.arange(0, 100, 100 / (s_in_train.shape[0]// samples_per_bin))
        bins = [np.percentile(s_in_train, p) for p in percentile_bins]
        binclass = np.digitize(s_in_test, bins)
        binclass_train = np.digitize(s_in_train, bins)
        avg_evolution_prediction = np.array([np.mean(s_out_train[binclass_train==b]) for b in binclass])

        return s_out_test, avg_evolution_prediction

    else:
        pass

### Metrics for Custom Plotting

def test_predictions(inputs):
    i_analogs = compute_best_analogs(inputs, inputs["max_analogs"])
    results = (
               np.mean(inputs["analog_output"][i_analogs]),
               np.min(inputs["analog_output"][i_analogs]),
               np.max(inputs["analog_output"][i_analogs]),
              )
    return results

def get_best_analog(inputs):
    i_analogs = compute_best_analogs(inputs, inputs["max_analogs"])
    results = inputs["analog_input"][i_analogs[0]]
    return results