#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 13:49:23 2017
History:
11/28/2020: modified for OSCAR 

@author: jaerock
"""


import sys
from drive_train_new_oscar_delay_mitig import DriveTrain
import gpu_options
from config_delay_mitig import Config
import datetime
import time

config = Config.neural_net
###############################################################################
#
def train(data_folder_name, base_model_path):
    gpu_options.set()
    
    drive_train = DriveTrain(data_folder_name, base_model_path)

    start_time = time.time()

    drive_train.train(show_summary = False)

    end_time = time.time()

    elapsed_seconds = end_time - start_time

    # Convert elapsed time to hours, minutes, and seconds
    elapsed_hours, remainder = divmod(elapsed_seconds, 3600)
    elapsed_minutes, elapsed_seconds = divmod(remainder, 60)

    print("Training time = {:02}h:{:02}m:{:02}s".format(int(elapsed_hours), int(elapsed_minutes), int(elapsed_seconds)))


###############################################################################
#
if __name__ == '__main__':
    try:
        if config['delay_mitig']==True:
            if len(sys.argv) !=3:
                exit('Delay Mitigation is True in the config file: please add the base model path')
            train(sys.argv[1],sys.argv[2]) # data_folder_path, base_model_path
        else:
            if (len(sys.argv) != 2):
                exit('Usage:\n$ python train.py data_path')

            train(sys.argv[1])

    except KeyboardInterrupt:
        print ('\nShutdown requested. Exiting...')
