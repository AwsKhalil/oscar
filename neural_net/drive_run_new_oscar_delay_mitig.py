#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 13:23:14 2017
History:
11/28/2020: modified for OSCAR 

@author: aws
"""
import sys
import cv2
import numpy as np
from net_model_new_oscar_delay_mitig import NetModel
from image_process import ImageProcess
from config_delay_mitig import Config

config = Config.neural_net
config_rn = Config.run_neural

def min_max_norm(in_data, in_max, in_min):
    scaled_data = (in_data-in_min)/(in_max-in_min)
    return scaled_data

###############################################################################
#
class DriveRun:
    
    ###########################################################################
    # model_path = 'path_to_pretrained_model_name' excluding '.h5' or 'json'
    # data_path = 'path_to_drive_data'  e.g. ../data/2017-09-22-10-12-34-56'
    def __init__(self, model_path, base_model_path):
        
        #self.config = Config()
        self.image_process = ImageProcess()
        self.net_model = NetModel(model_path, base_model_path)   
        self.net_model.load()

   ###########################################################################
    #
    def run(self, input): # input is (image, (vel))
        image = input[0]
        
        #The block below is commented out because the input image is already processed in 
        # NeuralControl
        #######################################################################################
        # image = cv2.resize(image, (Config.neural_net['input_image_width'],
        #                                 Config.neural_net['input_image_height']))
        # image = self.image_process.process(image)
        ######################################################################################

        npimg = np.expand_dims(image, axis=0)

        if Config.neural_net['num_inputs'] == 2:
            velocity = input[1]
            velocity = min_max_norm(velocity, Config.run_neural['max_vel'], 0.0)
            np_velocity = np.array(velocity).reshape(-1, 1)

        elif Config.neural_net['num_inputs'] == 3:
            velocity = input[1]
            velocity = min_max_norm(velocity, Config.run_neural['max_vel'], 0.0)
            np_velocity = np.array(velocity).reshape(-1, 1)
            acceleration = input[2]
            acceleration = min_max_norm(acceleration, 3.0, 0.0)
            np_acceleration = np.array(acceleration).reshape(-1,1)
        
        if Config.neural_net['num_inputs'] == 3:
            predict = self.net_model.model.predict([npimg, np_velocity, np_acceleration])
            if config_rn['delay']:
                predict_base = self.net_model.base_model.predict([npimg, np_velocity, np_acceleration])
        elif Config.neural_net['num_inputs'] == 2:
            predict = self.net_model.model.predict([npimg, np_velocity])
        else:
            predict = self.net_model.model.predict(npimg)

        # calc scaled steering angle
        steering_angle = predict[0][0]
        steering_angle /= config['steering_angle_scale']
        predict[0][0] = steering_angle

        if config_rn['delay']:    
            return predict, predict_base
        else:
            return predict


if __name__ == "__main__":
    try:
        if len(sys.argv) != 3:
            exit('Usage:\n$ python drive_run_new_oscar_delay_mitigation.py weight_file_name base_model_path')

        DriveRun(sys.argv[1], sys.argv[2]) # delay mitigation model path, base model path

    except KeyboardInterrupt:
        print ('\nShutdown requested. Exiting...')