import os
import base_directories

dir_settings = base_directories.get_directories()

for key in dir_settings:
    if key != "data_directory":
        mkdir_folder = dir_settings[key]
        print(mkdir_folder)
        os.system('mkdir ' + mkdir_folder)