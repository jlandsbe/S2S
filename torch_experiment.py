# Train the neural network approaches
import os
import regions
import importlib as imp
import numpy as np
import random
from pprint import pprint
import tensorflow as tf
import silence_tensorflow.auto
import build_data
import build_model
import experiments
from save_load_model_run import save_model_run
import train_model
import metrics
import pickle
import model_diagnostics_torch
import base_directories
import plots
import warnings
from torch_model import CustomDataset, TorchModel_base, TorchModel_gate, MaskTrainer, EarlyStopping, GateTrainer, prepare_device
from torch.utils.data import DataLoader
import torch
import torchinfo
from keras.utils.vis_utils import plot_model
warnings.filterwarnings(action='ignore', message='Mean of empty slice')

__author__ = "Jamin K. Rader, Elizabeth A. Barnes, and Randal J. Barnes"
__version__ = "30 March 2023"

dir_settings = base_directories.get_directories()



def gaussuian_filter(kernel_size, sigma=1, mu=0, noise = 0):
 
    # Initializing value of x,y as grid of kernel size
    # in the range of kernel size
 
    x, y = np.meshgrid(np.linspace(-1, 1, kernel_size),
                       np.linspace(-1, 1, kernel_size))
    dst = np.sqrt(x**2+y**2)
 
    # lower normal part of gaussian
    normal = 1/(2 * np.pi * sigma**2)
 
    # Calculating Gaussian filter
    gauss = np.exp(-((dst-mu)**2 / (2.0 * sigma**2))) * normal
    gauss = gauss + noise*np.random.rand(*gauss.shape)
    return gauss

def make_mjo_syn(settings, inpt, tsteps = 1000, ilat = 72, ilon = 144, olat = 12, olon = 144, speed = 36, mjo_strength = 32, mjo_size = 6,
                 noise_strength = 10, noise_size = 4, number_noise = 100, back_noise = .01, mjo_cycle_var = 0, enso=0):
    noise_gauss = []
    memory = "neutral"
    input_arr = back_noise*(np.random.rand(tsteps, ilat, ilon)-.5)
    output_arr = np.zeros((tsteps, olat, olon))
    prev_maps = np.zeros((tsteps, olat, olon))
    mjo = mjo_strength*gaussuian_filter(mjo_size)
    for num in range(number_noise):
        ns = np.random.choice([1])*noise_strength*gaussuian_filter(noise_size)
        ns = ns + (np.random.choice([1])*noise_strength*(.2*np.random.rand())*gaussuian_filter(noise_size))
        noise_gauss.append(ns)
    start_row = (output_arr.shape[1] - mjo.shape[0]) // 2
    output_arr[0,0:mjo.shape[0], 0: mjo.shape[1]] += mjo
    prev_maps[0,0:mjo.shape[0], 0: mjo.shape[1]] += mjo
    for i in range(1,tsteps):
        output_arr[i,:,:] = np.roll(prev_maps[i-1,:,:], speed, axis=1)
        prev_maps[i,:,:] = np.roll(prev_maps[i-1,:,:], speed, axis=1)
        if enso>0:
            if not i%int(1*ilon/speed):
                #this should really be 6*ilon, but I'm doing 1 so we see it more
                phase = np.random.rand()
                if phase >.8:
                    memory = "nino"
                elif phase <.2:
                    memory = "nina"
                else:
                    memory = "neutral"
        if memory == "nino":
            output_arr[i,2:olat-2,65:110] = output_arr[i,2:olat-2,65:110] + enso + .1*(np.random.rand(*(output_arr[i,2:olat-2,65:110].shape))-.5)
        if memory == "nina":
            output_arr[i,2:olat-2,65:110] = output_arr[i,2:olat-2,65:110] - enso + .1*(np.random.rand(*(output_arr[i,2:olat-2,65:110].shape))-.5)
        
        if not (int(i-mjo_size/2)%(144/speed)) and mjo_cycle_var>0 and mjo_cycle_var<1:
            xlow = 1/(1+mjo_cycle_var)
            xhigh = 1+mjo_cycle_var
            random_float = random.uniform(xlow, xhigh)
            output_arr[i] = output_arr[i] * random_float
            prev_maps[i] = prev_maps[i]* random_float
    for arr in input_arr:
        for ng in noise_gauss:
            x_pos = np.random.randint(0, arr.shape[0] - ng.shape[0] + 1)
            y_pos = np.random.randint(0, arr.shape[1] - ng.shape[1] + 1)
            arr[x_pos:x_pos+ng.shape[0], y_pos:y_pos+ng.shape[1]] = ng
    clean_output = np.copy(output_arr)
    for arr2 in output_arr:
        for count, ng in enumerate(noise_gauss):
            if count%6 == 0:
                x_pos = np.random.randint(0, arr2.shape[0] - ng.shape[0] + 1)
                y_pos = np.random.randint(0, arr2.shape[1] - ng.shape[1] + 1)
                arr2[x_pos:x_pos+ng.shape[0], y_pos:y_pos+ng.shape[1]] += .5*ng
    insert_row = (input_arr.shape[1] - output_arr.shape[1]) // 2
    input_arr[:,insert_row:insert_row + output_arr.shape[1], :] += clean_output  

    #input_arr = (input_arr - mean_value)/std_value
    #output_arr = (output_arr - mean_value)/std_value
    #output_arr = input_arr[:,insert_row:insert_row + output_arr.shape[1], :]
    if inpt:
        input_arr = input_arr[:,:,:,np.newaxis]
        return input_arr[:-settings["lead_time"]]
    else:
        return output_arr[settings["lead_time"]:]
    
def get_masks_for_assess(soi_ins, soi_out):
    #masks = np.zeros(np.shape(soi_ins[:,:,:,0]))
    masks = np.zeros(np.shape(soi_ins[:,:,:]))
    mjo = 1*gaussuian_filter(6)
    mjo = mjo/mjo
    output_arr = np.zeros(np.shape(soi_out))
    #output_arr[0,0:mjo.shape[0], 0: mjo.shape[1]] += mjo
    output_arr[0:mjo.shape[0], 0: mjo.shape[1]] += mjo
    insert_row = (masks.shape[1] - output_arr.shape[0]) // 2
    masks[:,insert_row:insert_row + output_arr.shape[0], :] += output_arr  
    for i in range(1,len(masks)):
        masks[i,:,:] = np.roll(masks[i-1,:,:], 36, axis=1)
    return masks

def JBL_assess(masks, soi_ins, analog_ins, soi_outs, analog_outs, m = 0, c = 0.0):
    soi_ins = soi_ins[:,:,:,0]
    analog_ins = analog_ins[:,:,:,0] #just removing added axis for extra channels
    A = np.array([])
    B = np.array([])
    for i in range(len(masks)):
        weighted_soi_in = soi_ins[i]*masks[i]
        weighted_analog_ins = analog_ins * masks[i]
        all_ins_mae = np.mean((weighted_soi_in-weighted_analog_ins)**2, axis=(1,2))
        all_outs_mae = np.mean((soi_outs[i]-analog_outs)**2, axis=(1,2))
        #now I need to do the linear transform to get them as close as possible
        A  = np.append(A,all_ins_mae)
        B  = np.append(B, all_outs_mae)
    if m==0:
        m = (np.sum(A*B) - np.mean(B)*np.sum(A))/(np.sum(A*A)-np.mean(A)*np.sum(A))
        c = np.mean(B) - m*np.mean(A)
    A_tr = m*A + c
    diffs = np.mean((A_tr-B)**2)
    print("Weights (m):", m)
    print("Biases (c):", c)
    return np.mean(diffs)

def train_experiments(
    exp_name_list,
    data_directory,
    model_path,
    overwrite_model=False,
    base_exp_name = None,
    settings_overwrite = None
):
    for exp_name in exp_name_list:
        
        settings = experiments.get_experiment(exp_name, base_exp_name=base_exp_name,
                                              settings_overwrite=settings_overwrite)

        print('-- TRAINING ' + settings["exp_name"] + ' --')
            
        (
            analog_input,
            analog_output,
            soi_train_input,
            soi_train_output,
            soi_val_input,
            soi_val_output,
            soi_test_input,
            soi_test_output,
            input_standard_dict,
            output_standard_dict,
            lat,
            lon,
            persist_err,
        ) = build_data.build_data(settings, data_directory)
        if settings["total_synthetic"]:
            analog_input = make_mjo_syn(settings, 1, 2500)
            analog_output = make_mjo_syn(settings, 0, 2500)
            soi_train_input = make_mjo_syn( settings, 1, 2500)
            soi_train_output = make_mjo_syn(settings, 0, 2500)
            soi_val_input = make_mjo_syn(settings, 1,1000)
            soi_val_output = make_mjo_syn(settings, 0,1000)
            soi_test_input = make_mjo_syn(settings, 1,1000)
            soi_test_output = make_mjo_syn(settings, 0,1000)
        for rng_seed in settings["rng_seed_list"]:  #0 to 100 by 10s
            settings["rng_seed"] = rng_seed

            for model_type in settings["model_type_list"]:
                settings["model_type"] = model_type
                print(model_type)
                # Create the model name.
                savename_prefix = (
                        settings["exp_name"]
                        + "_" + settings["model_type"] + "_"
                        + f"rng_seed_{settings['rng_seed']}"
                )
                settings["savename_prefix"] = savename_prefix
                print('--- RUNNING ' + savename_prefix + '---')

                # Check if the model metrics exist and overwrite is off.
                metric_savename = dir_settings["metrics_directory"]+settings["savename_prefix"]+'_subset_metrics.pickle'
                if os.path.exists(metric_savename) and overwrite_model is False:
                    print(f"   saved {settings['savename_prefix']} metrics already exist. Skipping...")
                    continue
                if settings["gif"]:
                    output_plot = analog_input*np.nan
                    insert_row = (analog_input.shape[1] - analog_output.shape[1]) // 2
                    output_plot[:,insert_row:insert_row + analog_output.shape[1], :, :] = analog_output[:,:,:,np.newaxis]
                    model_diagnostics_torch.video_syn_data(settings,analog_input[0:200,:,:,:],lat,lon, sv= "_input_")
                    model_diagnostics_torch.video_syn_data(settings,output_plot[0:200,:,:,:],lat,lon, sv = "_output_")
                # Make, compile, train, and save the model.
                tf.keras.backend.clear_session()
                np.random.seed(settings["rng_seed"])
                random.seed(settings["rng_seed"])
                tf.random.set_seed(settings["rng_seed"])
                print(settings["model_type"])
                if settings["model_type"] == "ann_analog_model":
                    model = build_model.build_ann_analog_model(
                        settings, [soi_train_input, analog_input])
                elif settings["model_type"] == "ann_model":
                    model = build_model.build_ann_model(
                        settings, [analog_input])
                elif settings["model_type"] == "interp_model":

                ###this is where I would create data, create new torch model
                    trainset = CustomDataset(analog_input, soi_train_input, analog_output, soi_train_output)
                    valset = CustomDataset(analog_input, soi_val_input, analog_output, soi_val_output)
                    train_loader = DataLoader(
                    trainset,
                    batch_size=settings["batch_size"],
                    shuffle=True,
                    drop_last=False,
                )
                    val_loader = DataLoader(
                    valset,
                    batch_size=settings["val_batch_size"],
                    shuffle=True,
                    drop_last=False,
)
                    model = TorchModel_base(settings, np.shape(soi_train_input)[1:])
                    criterion = torch.nn.MSELoss()
                    optimizer = torch.optim.Adam(model.parameters(), lr=settings["interp_learning_rate"])
                    device = prepare_device()
                    metric_funcs = []
                    trainer = MaskTrainer(
                        model,
                        criterion,
                        metric_funcs,
                        optimizer,
                        max_epochs=settings["max_epochs"],
                        data_loader=train_loader,
                        validation_data_loader=val_loader,
                        device=device,
                        settings=settings,
)
                    torchinfo.summary(
    model,
    [
        analog_input[: settings["batch_size"]].shape,
        soi_train_input[: settings["batch_size"]].shape,
    ],
    verbose=1,
    col_names=("input_size", "output_size", "num_params"),
)

                    model.to(device)
                    trainer.fit()
                else:
                     raise NotImplementedError("no such model coded yet")
                if settings["model_type"] == "interp_model":
                    if settings["state_masks"] == 1:
                            (weights_val, dissimilarities_val, prediction_val, gates, branches) = build_model.parse_model(x_val, model.layers[2], model.layers[3], model.layers[4],) #check on whether these are the right model layers
                            #xval is 2 x val_batch x lat x lon x channel
                            #check weights_val
                            mean_weights_val = np.mean(weights_val, axis=0)[np.newaxis, :, :, :]
                            for i in range(0,len(weights_val), int(len(weights_val)/5)):
                                nm = "_combined_mask_" + str(i)
                                nm2 = "_gate_" + str(np.argmax(gates[i])) +  "_branches_" + str(i)
                                nm3 = "_soi_sample_" + str(i) + "_gates_" + str(np.mean(gates,axis=0))
                                #model_diagnostics.visualize_interp_model(settings, weights_val[i], lat, lon, sv = nm, clims = (round((np.mean(np.min(weights_val, axis=(1, 2)))),6), round(np.mean(np.max(weights_val, axis=(1, 2))),6)))
                                model_diagnostics_torch.visualize_interp_model(settings, x_val[0][i], lat, lon, sv = nm3)
                                model_diagnostics_torch.visualize_interp_model(settings, branches[i], lat, lon, sv = nm2, ttl = gates[i])
                                print("done")
                            #weights_val = (2500, 72, 144, 1), 2500 coming from validation batch size
                    #this returns the mask! So could cut it here to just get the mask
                    else:
                        map_layer = getattr(model, "bias_only")
                        biases = map_layer.data
                        weights_val = biases.numpy().reshape(np.shape(analog_input)[1:])
                        if settings["gates"] or 1:
                            reg_map = np.zeros(np.shape(weights_val))
                            reg_map = build_data.extract_region(reg_map, regions.get_region_dict(settings["target_region_name"]), lat=lat, lon=lon, mask_builder = 1)
                            map_options = np.stack([weights_val, reg_map])
                            #Here is where I can run the network
                            gate_model = TorchModel_gate(settings, np.shape(soi_train_input)[1:], map_options)
                            criterion = torch.nn.MSELoss()
                            optimizer = torch.optim.Adam(model.parameters(), lr=settings["interp_learning_rate"])
                            device = prepare_device()
                            metric_funcs = []
                            trainer = MaskTrainer(
                                gate_model,
                                criterion,
                                metric_funcs,
                                optimizer,
                                max_epochs=settings["max_epochs"],
                                data_loader=train_loader,
                                validation_data_loader=val_loader,
                                device=device,
                                settings=settings,
        )
                            model.to(device)
                            trainer.fit()

                        model_diagnostics_torch.visualize_interp_model(settings, weights_val, lat, lon)
                else:
                    weights_val = None
                #this is where I will do some evaluation
                #first I will pass in my soi_test_input to parse_model to get 238 weights out
                #then for each of these soi_test_inputs I will multiply the mask by the one soi_test_input and by all the possible analogs (could use analog_inputs)
                #then for each soi_test_inputs I will also find the difference in the same indexed soi_test_output and each of the analog_outputs
                #Then I will compare all the inputs with outputs (MAE?) and lastly take the mean
                #I will repeat the process above, but with my own mask
                #This mask will be found by finding the brightest spot on the soi_test_input and making a box around it of size mjo_shape
                #or actually since this is in order, I can just use the same starting point and roll my mask along, just like I do in my build mjo data
                #I will do this for all SOIs and those will be my new 238 weights out
                #then I repeat the process
                #compare end results


                # PLOT MODEL EVALUATION METRICS
                if settings["state_masks"] == 1:
                    weights_val = None 
                if settings["gates"] or 1:
                    metrics_dict = model_diagnostics_torch.visualize_metrics(settings, model, soi_test_input, soi_test_output,
                                                                    analog_input, analog_output, lat,
                                                                    lon, weights_val, persist_err,
                                                                    n_testing_analogs=analog_input.shape[0],
                                                                    analogue_vector = settings["analogue_vec"],
                                                                    soi_train_output = soi_train_output,
                                                                    fig_savename="subset_skill_score_vs_nanalogues", gates=gates)
                else:
                    metrics_dict = model_diagnostics_torch.visualize_metrics(settings, model, soi_test_input, soi_test_output,
                                                                    analog_input, analog_output, lat,
                                                                    lon, weights_val, persist_err,
                                                                    n_testing_analogs=analog_input.shape[0],
                                                                    analogue_vector = settings["analogue_vec"],
                                                                    soi_train_output = soi_train_output,
                                                                    fig_savename="subset_skill_score_vs_nanalogues")

                # SAVE THE METRICS
                print("almost at the end")
                with open(dir_settings["metrics_directory"]+settings["savename_prefix"]
                          + '_subset_metrics.pickle', 'wb') as f:
                    pickle.dump(metrics_dict, f, protocol=pickle.HIGHEST_PROTOCOL)


