# Train the neural network approaches
import os
import regions
import importlib as imp
import numpy as np
import random
from pprint import pprint
import tensorflow as tf
import silence_tensorflow.auto


#import build_model
import experiments
from save_load_model_run import save_model_run
import train_model
import metrics
import pickle
import model_diagnostics_torch
import base_directories
import plots
import warnings
from torch_model import CustomDataset, TorchModel_base, TorchModel_gate,TorchModel_gatedMap,CombinedGateModel,MaskTrainer, EarlyStopping, GateTrainer, prepare_device
from torch.utils.data import DataLoader
import torch
import torchinfo
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt
warnings.filterwarnings(action='ignore', message='Mean of empty slice')
warnings.filterwarnings("ignore", message="Failed to load image Python extension")

__author__ = "Jamin K. Rader, Elizabeth A. Barnes, and Randal J. Barnes"
__version__ = "30 March 2023"

dir_settings = base_directories.get_directories()


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
        if settings["monthly_data"]:
            import build_data_monthly as build_data
        else:
            import build_data

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
            analog_dates,
            soi_dates, tether_analogs, tether_soi, progression_analogs, progression_soi
        ) = build_data.build_data(settings, data_directory)
        if settings["exp_name"] == 'natl_ext_temp_wind_to_wind_winter_m1_obs_test':
            soi_train_input = soi_test_input
            soi_train_output = soi_test_output
        if settings["tethers"]==[]:
            tether_analogs = []
            tether_soi = []
        if settings["preprocess"]:
            print("Preprocessing completed")
        else:
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

                    # Make, compile, train, and save the model.
                    tf.keras.backend.clear_session()
                    np.random.seed(settings["rng_seed"])
                    random.seed(settings["rng_seed"])
                    tf.random.set_seed(settings["rng_seed"])
                    print(settings["model_type"])
                    if settings["model_type"] == "ann_analog_model":
                        # model = build_model.build_ann_analog_model(
                        #     settings, [soi_train_input, analog_input])
                        pass
                    elif settings["model_type"] == "ann_model":
                        # model = build_model.build_ann_model(
                        #     settings, [analog_input])
                        pass
                    elif settings["model_type"] == "interp_model" and not settings["presaved_mask"]:
                        trainset = CustomDataset(analog_input, soi_train_input, analog_output, soi_train_output, tether_analogs, tether_soi)
                        valset = CustomDataset(analog_input, soi_val_input, analog_output, soi_val_output, tether_analogs, tether_soi, np.random.randint(0, len(soi_val_input), size=min(settings["max_iterations"], len(analog_input))), np.random.randint(0, len(analog_input), size=min(settings["max_iterations"], len(analog_input))))
                        train_loader = DataLoader(
                        trainset,
                        batch_size=settings["batch_size"],
                        shuffle=False,
                        drop_last=False
                    )
                        val_loader = DataLoader(
                        valset,
                        batch_size=settings["val_batch_size"],
                        shuffle=False,
                        drop_last=False
    )
                        model = TorchModel_base(settings, np.shape(soi_train_input)[1:])
                        criterion = torch.nn.MSELoss(reduction='none')
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
                        trainer.plot_loss()
                        model_savename = dir_settings["model_directory"] + savename_prefix + "_torch_model"
                        torch.save(model.state_dict(),model_savename)
                    elif type(settings["presaved_mask"]) == type("string"):
                        model = TorchModel_base(settings, np.shape(soi_train_input)[1:])
                        mask_save_name = (settings["presaved_mask"]+ "_" + settings["model_type"] + "_"+ f"rng_seed_{settings['rng_seed']}")
                        model_savename = dir_settings["model_directory"] + mask_save_name + "_torch_model"
                        model.load_state_dict(torch.load(model_savename))
                    elif settings["presaved_mask"]:
                        model = TorchModel_base(settings, np.shape(soi_train_input)[1:])
                        model_savename = dir_settings["model_directory"] + savename_prefix + "_torch_model"
                        model.load_state_dict(torch.load(model_savename))
                    else:
                        raise NotImplementedError("no such model coded yet")
                    if settings["model_type"] == "interp_model":
                            map_layer = getattr(model, "bias_only")
                            biases = map_layer.data
                            weights_val = biases.numpy().reshape(np.shape(analog_input)[1:])
                            if settings["cutoff"]>0:
                                if settings["cutoff"]>1:
                                    weights_val = weights_val**settings["cutoff"]
                                    weights_val = weights_val/np.mean(weights_val)
                                else:
                                    weights_val = np.where(weights_val>=np.quantile(weights_val,settings["cutoff"]), weights_val, 0)
                            if len(settings["ablation"])>0:
                                for ab_region in settings["ablation"]:
                                    #randomly zero out points (number of points equal to number of points in the region)
                                    flag = 1
                                    if len(ab_region)>7:
                                        a_chunk1 = ab_region[0:7]
                                        a_chunk2 = ab_region[7:]
                                        if a_chunk1 == "random_":
                                            flag = 0
                                            regn = regions.get_region_dict(a_chunk2)
                                            min_lon, max_lon = regn["lon_range"]
                                            min_lat, max_lat = regn["lat_range"]
                                            ilon = np.where((lon >= min_lon) & (lon <= max_lon))[0]
                                            ilat = np.where((lat >= min_lat) & (lat <= max_lat))[0]
                                            num_points = len(ilon) * len(ilat)
                                            # Generate d random indices for the first two dimensions
                                            m,n,p = np.shape(weights_val)
                                            random_indices = np.random.choice(m * n, size=num_points, replace=False)
                                            rows, cols = np.unravel_index(random_indices, (m, n))
                                            # Set the same random points to zero in all layers of the third dimension
                                            weights_val[rows, cols, :] = 0
                                    if ab_region == "ocean":    
                                        weights_val = build_data.mask_in_land_ocean(weights_val, settings, "land")
                                    elif ab_region == "land":
                                        weights_val = build_data.mask_in_land_ocean(weights_val, settings, "ocean")
                                    elif ab_region == "ocean_temp":
                                        weights_val[:,:,0] = build_data.mask_in_land_ocean(weights_val, settings, "land")[:,:,0]
                                    elif ab_region == "land_temp":
                                        weights_val[:,:,0] = build_data.mask_in_land_ocean(weights_val, settings, "ocean")[:,:,0]
                                    elif ab_region == "temp":
                                        weights_val[:,:,0] = 0
                                    elif ab_region == "u250":
                                        weights_val[:,:,-1] = 0
                                    elif flag:
                                        if ab_region == "jet_stream":
                                            only_u = 1
                                            weights_val, full_weights = build_data.extract_region(weights_val, region = regions.get_region_dict(ab_region), lat = lat, lon=lon, ablation = 1, only_u = only_u)
                                        else:
                                            weights_val, full_weights = build_data.extract_region(weights_val, region = regions.get_region_dict(ab_region), lat = lat, lon=lon, ablation = 1)

                            if settings["gates"]:
                                reg_map_na = np.zeros(np.shape(weights_val))
                                reg_map_na = build_data.extract_region(reg_map_na, regions.get_region_dict("n_atlantic"), lat=lat, lon=lon, mask_builder = 1)
                                reg_map_np = np.zeros(np.shape(weights_val))
                                reg_map_np = build_data.extract_region(reg_map_np, regions.get_region_dict("n_pacific"), lat=lat, lon=lon, mask_builder = 1)
                                map_options = np.stack([reg_map_na, reg_map_np])
                                model_diagnostics_torch.visualize_interp_model(settings, np.squeeze(np.stack([reg_map_na, reg_map_np], axis=-1)), lat, lon)
                                #Here is where I can run the network
                                gate_model = CombinedGateModel(settings, np.shape(soi_train_input)[1:], map_options)
                                criterion = torch.nn.MSELoss()
                                optimizer = torch.optim.Adam(model.parameters(), lr=settings["interp_learning_rate"])
                                device = prepare_device()
                                metric_funcs = []
                                trainer = GateTrainer(
                                    gate_model,
                                    criterion,
                                    metric_funcs,
                                    optimizer,
                                    max_epochs=settings["gate_max_epochs"],
                                    data_loader=train_loader,
                                    validation_data_loader=val_loader,
                                    device=device,
                                    settings=settings,
            )
                                gate_model.to(device)
                                trainer.fit()

                                gate_model.eval()
                                with torch.no_grad():
                                    soi_test_input_tensor = torch.from_numpy(soi_test_input).float()
                                    gate_tensor = gate_model.gate_model(soi_test_input_tensor)
                                    gates = gate_tensor.cpu().numpy()
                                plt.hist(np.argmax(gates,axis=-1), align='mid', bins=[-.25,.25,.75,1.25], edgecolor='black', linewidth=1.2, color='cornflowerblue')
                                plt.title("Gate Distribution: 0 = NA, 1 = NP")
                                plt.savefig(dir_settings["figure_directory"] + settings["savename_prefix"] +
                        '_gate_distribution.png', dpi=300, bbox_inches='tight')
                        
                            model_diagnostics_torch.visualize_interp_model(settings, weights_val, lat, lon)
                    else:
                        weights_val = None


                    if settings["mask_only"]:
                        continue
                    # PLOT MODEL EVALUATION METRICS
                    if settings["gates"]:
                        metrics_dict, crps_dict = model_diagnostics_torch.visualize_metrics(settings, model, soi_test_input, soi_test_output,
                                                                        analog_input, analog_output, np.array(progression_analogs), np.array(progression_soi), lat,
                                                                        lon, weights_val, persist_err,
                                                                        n_testing_soi=int(soi_test_input.shape[0]*settings["percent_soi"]),
                                                                        n_testing_analogs=int(analog_input.shape[0]*settings["percent_analog"]),
                                                                        analogue_vector = settings["analogue_vec"],
                                                                        soi_train_output = None,
                                                                        fig_savename="subset_skill_score_vs_nanalogues", gates=gates, percentile = settings["extremes_percentile"])
                    else:
                        metrics_dict, crps_dict = model_diagnostics_torch.visualize_metrics(settings, model, soi_test_input, soi_test_output,
                                                                        analog_input, analog_output, np.array(progression_analogs), np.array(progression_soi), lat,
                                                                        lon, weights_val, persist_err,
                                                                        n_testing_soi=int(soi_test_input.shape[0]*settings["percent_soi"]),
                                                                        n_testing_analogs=int(analog_input.shape[0]*settings["percent_analog"]),
                                                                        analogue_vector = settings["analogue_vec"],
                                                                        soi_train_output = None,
                                                                        fig_savename="subset_skill_score_vs_nanalogues", analog_dates = analog_dates, soi_dates = soi_dates, percentile = settings["extremes_percentile"])

                    # SAVE THE METRICS
                    print("almost at the end")
                    with open(dir_settings["metrics_directory"]+settings["savename_prefix"]
                            + '_subset_metrics.pickle', 'wb') as f:
                        pickle.dump(metrics_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

                    with open(dir_settings["metrics_directory"]+settings["savename_prefix"]
                            + '_crps_only_metrics.pickle', 'wb') as f:
                        pickle.dump(crps_dict, f, protocol=pickle.HIGHEST_PROTOCOL)


