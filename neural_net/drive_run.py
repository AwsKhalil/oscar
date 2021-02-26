#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 13:23:14 2017
History:
11/28/2020: modified for OSCAR 

@author: jaerock
"""

import numpy as np
from net_model import NetModel
from config import Config

###############################################################################
#
class DriveRun:
    
    ###########################################################################
    # model_path = 'path_to_pretrained_model_name' excluding '.h5' or 'json'
    # data_path = 'path_to_drive_data'  e.g. ../data/2017-09-22-10-12-34-56'
    def __init__(self, model_path):
        
        #self.config = Config()
        self.net_model = NetModel(model_path)   
        self.net_model.load()

   ###########################################################################
    #
    def run(self, image, vel):
        np_img = np.expand_dims(image, axis=0)
        np_img = np.array(np_img).reshape(-1, 
                                          Config.neural_net['input_image_height'],
                                          Config.neural_net['input_image_width'],
                                          Config.neural_net['input_image_depth'])
        
        if Config.neural_net['train_velocity'] is True:
            vel = np.array(vel).reshape(-1, 1)
            measurements = self.net_model.model.predict([np_img, vel])
            throttle = measurements[0][1]
            steering_angle = measurements[0][0] / Config.neural_net['steering_angle_scale']
            return steering_angle, throttle
        else:
            measurements = self.net_model.model.predict(np_img)
            measurements = measurements / Config.neural_net['steering_angle_scale']
            return measurements
