# DRIVE THE TRAIN
# Train the neural network approaches

import time
import numpy as np
import tensorflow as tf
from torch_experiment import train_experiments
import base_directories

#tf.config.set_visible_devices([], "GPU")  # turn-off tensorflow-metal if it is on
#np.warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

__author__ = "Jamin K. Rader, Elizabeth A. Barnes, and Randal J. Barnes"
__version__ = "30 March 2023"

# List of experiments to run
#EXP_NAME_LIST = ("natl_ext_temp_wind_to_wind_winter_m12_obs_tercile_ablation_land","natl_ext_temp_wind_to_wind_winter_m12_obs_tercile_ablation_SH","natl_ext_temp_wind_to_wind_winter_m12_obs_tercile_ablation_npac","natl_ext_temp_wind_to_wind_winter_m12_obs_tercile_switch_mask", "natl_ext_temp_wind_to_wind_winter_m1_obs_tercile_switch_mask","natl_ext_temp_wind_to_wind_winter_m1_tercile_switch_mask","natl_ext_temp_wind_to_wind_winter_m12_tercile_switch_mask")
#EXP_NAME_LIST = ("natl_ext_temp_wind_to_wind_winter_m12_obs_tercile_ablation_80_cutoff","natl_ext_temp_wind_to_wind_winter_m12_tercile_ablation_80_cutoff","natl_ext_temp_wind_to_wind_winter_m12_obs_tercile_ablation_40_cutoff","natl_ext_temp_wind_to_wind_winter_m12_tercile_ablation_40_cutoff","natl_ext_temp_wind_to_wind_winter_m12_obs_tercile_ablation_20_cutoff","natl_ext_temp_wind_to_wind_winter_m12_tercile_ablation_20_cutoff",)
#EXP_NAME_LIST = ( "natl_ext_temp_wind_to_wind_winter_m12_obs_tercile_ablation_natl","natl_ext_temp_wind_to_wind_winter_m12_obs_tercile_switch_mask",)
# EXP_NAME_LIST = ("natl_ext_temp_wind_to_wind_winter_m12_tercile_extra",
#     "natl_ext_temp_wind_to_wind_winter_m12_tercile_ablation_temp_extra",
#     "natl_ext_temp_wind_to_wind_winter_m12_tercile_ablation_u250_extra",
#     "natl_ext_temp_wind_to_wind_winter_m12_tercile_ablation_SH_extra",
#     "natl_ext_temp_wind_to_wind_winter_m12_tercile_ablation_natl_extra",
#     "natl_ext_temp_wind_to_wind_winter_m12_tercile_ablation_npac_extra",
#     "natl_ext_temp_wind_to_wind_winter_m12_tercile_switch_mask_extra",
#     "natl_ext_temp_wind_to_wind_winter_m12_tercile_ablation_20_cutoff_extra",
#     "natl_ext_temp_wind_to_wind_winter_m12_tercile_ablation_40_cutoff_extra",
#     "natl_ext_temp_wind_to_wind_winter_m12_tercile_ablation_80_cutoff_extra",
#     "natl_ext_temp_wind_to_wind_winter_m12_tercile_ablation_90_cutoff_extra",
#     "natl_ext_temp_wind_to_wind_winter_m12_obs_tercile_extra",
#     "natl_ext_temp_wind_to_wind_winter_m12_obs_tercile_ablation_temp_extra",
#     "natl_ext_temp_wind_to_wind_winter_m12_obs_tercile_ablation_u250_extra",
#     "natl_ext_temp_wind_to_wind_winter_m12_obs_tercile_ablation_SH_extra",
#     "natl_ext_temp_wind_to_wind_winter_m12_obs_tercile_ablation_natl_extra",
#     "natl_ext_temp_wind_to_wind_winter_m12_obs_tercile_ablation_npac_extra",
#     "natl_ext_temp_wind_to_wind_winter_m12_obs_tercile_switch_mask_extra",
#     "natl_ext_temp_wind_to_wind_winter_m12_obs_tercile_ablation_20_cutoff_extra",
#     "natl_ext_temp_wind_to_wind_winter_m12_obs_tercile_ablation_40_cutoff_extra",
#     "natl_ext_temp_wind_to_wind_winter_m12_obs_tercile_ablation_80_cutoff_extra",
#     "natl_ext_temp_wind_to_wind_winter_m12_obs_tercile_ablation_90_cutoff_extra")

# EXP_NAME_LIST = ("natl_ext_temp_wind_to_wind_winter_m12_obs_tercile_ablation_random_npac_extra","natl_ext_temp_wind_to_wind_winter_m12_tercile_ablation_random_npac_extra","natl_ext_temp_wind_to_wind_winter_m12_obs_tercile_ablation_random_natl_extra","natl_ext_temp_wind_to_wind_winter_m12_tercile_ablation_random_natl_extra","natl_ext_temp_wind_to_wind_winter_m12_obs_tercile_ablation_random_SH_extra","natl_ext_temp_wind_to_wind_winter_m12_tercile_ablation_random_SH_extra")

# EXP_NAME_LIST = ("natl_ext_temp_wind_to_wind_winter_m12_obs_tercile", "natl_ext_temp_wind_to_wind_winter_m12_tercile")
EXP_NAME_LIST = ("natl_ext_temp_wind_to_wind_winter_m12_tercile_extra", "natl_ext_temp_wind_to_wind_winter_m12_tercile_ablation_ocean_temp","natl_ext_temp_wind_to_wind_winter_m12_tercile_ablation_land_temp", "natl_ext_temp_wind_to_wind_winter_m12_tercile_ablation_80_cutoff_extra",
"natl_ext_temp_wind_to_wind_winter_m12_tercile_ablation_90_cutoff_extra","natl_ext_temp_wind_to_wind_winter_m12_tercile_ablation_40_cutoff_extra",
"natl_ext_temp_wind_to_wind_winter_m12_tercile_network_test","natl_ext_temp_wind_to_wind_winter_m12_tercile_ablation_temp_extra",
"natl_ext_temp_wind_to_wind_winter_m12_tercile_ablation_u250_extra",
)
EXP_NAME_LIST = ("california_temp_to_temp_week_summer_90th_high_low","california_temp_to_temp_week_summer_90th_obs_high_low","california_temp_to_temp_week34_summer_tercile","california_temp_to_temp_week34_obs_summer_tercile")


if __name__ == "__main__":

    start_time = time.time()

    dir_settings = base_directories.get_directories()

    train_experiments(
        EXP_NAME_LIST,
        dir_settings["data_directory"],
        dir_settings["model_directory"],
        overwrite_model=True,
    )

    elapsed_time = time.time() - start_time
    #print(f"Total elapsed time: {elapsed_time:.2f} seconds")
