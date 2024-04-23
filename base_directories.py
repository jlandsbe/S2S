""" Define default settings."""

import socket
import os

def get_data_directory():
    data_directory = '/Users/jlandsbe/Downloads/WeightedMaskAnalogForecasting-main/Libby_23/tiniest_test'

    return data_directory


def get_directories():

    data_directory = get_data_directory()

    dir_settings = {
        "data_directory": data_directory + '/data/',
        "net_data": os.path.dirname(data_directory),
        "example_data_directory" : data_directory + '/example_data/',
        "figure_directory" : data_directory + '/figures/',
        "figure_diag_directory": data_directory + '/figures/model_diagnostics/',
        "figure_metrics_directory": data_directory + '/figures/metrics_summary/',
        "figure_custom_directory": data_directory + '/figures/custom/',
        "model_directory":data_directory + '/saved_models/',
        "metrics_directory":data_directory + '/saved_metrics/',
        "tuner_directory" :data_directory + '/tuning_results/',
        "figure_tuner_directory" :data_directory + '/figures/tuning_figures/',
        "tuner_autosave_directory":data_directory + '/tuning_results/autosave/'

    }

    return dir_settings
