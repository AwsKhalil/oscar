#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 22:07:31 2019
History:
11/28/2020: modified for OSCAR 

@author: jaerock
"""

import cv2
import numpy as np
#import keras
#import sklearn
#import resnet
from progressbar import ProgressBar
import matplotlib.pyplot as plt

import const
from net_model_new_oscar_delay_mitig import NetModel
from drive_data_new_oscar_delay_mitig import DriveData
from config_delay_mitig import Config
from image_process import ImageProcess

###############################################################################
#

config = Config.neural_net
config_dc = Config.data_collection

def min_max_scaling(data):
    min_val = np.min(data)
    max_val = np.max(data)
    scaled_data = (data - min_val) / (max_val - min_val)
    return scaled_data

def is_min_max_normalized(data):
    return np.all((data >= 0) & (data <= 1))

class DriveLog:
    
    ###########################################################################
    # model_path = 'path_to_pretrained_model_name' excluding '.h5' or 'json'
    # data_path = 'path_to_drive_data'  e.g. ../data/2017-09-22-10-12-34-56'
       
    def __init__(self, base_model_path, model_path, data_path):
        if data_path[-1] == '/':
            data_path = data_path[:-1]

        loc_slash = data_path.rfind('/')
        if loc_slash != -1: # there is '/' in the data path
            model_name = data_path[loc_slash+1:] # get folder name
            #model_name = model_name.strip('/')
        else:
            model_name = data_path

        csv_path = data_path + '/' + model_name + '_delay_mitigation' + const.DATA_EXT   
        
        self.data_path = data_path
        self.data = DriveData(csv_path)
        self.data = self.data
        self.test_generator = None
        
        self.num_test_samples = 0        
        #self.config = Config()
        
        self.net_model = NetModel(model_path, base_model_path)
        self.net_model.load()
        self.model_path = model_path
        
        self.image_process = ImageProcess()

        # delta 0
        self.steering0_measurements = []
        self.steering0_predictions = []
        self.steering0_differences = []
        self.steering0_squared_differences = []

        # delta 9
        self.steering9_measurements = []
        self.steering9_predictions = []
        self.steering9_differences = []
        self.steering9_squared_differences = []

        # delta 12
        self.steering12_measurements = []
        self.steering12_predictions = []
        self.steering12_differences = []
        self.steering12_squared_differences = []

        # delta 15
        self.steering15_measurements = []
        self.steering15_predictions = []
        self.steering15_differences = []
        self.steering15_squared_differences = []

        # delta 18
        self.steering18_measurements = []
        self.steering18_predictions = []
        self.steering18_differences = []
        self.steering18_squared_differences = []

        # delta 21
        self.steering21_measurements = []
        self.steering21_predictions = []
        self.steering21_differences = []
        self.steering21_squared_differences = []

    ###########################################################################
    #
    def _prepare_data(self):
        
        self.data.read(normalize = False)

        # Normalize velocity and acceleration
        self.data.velocities = min_max_scaling(self.data.velocities)
        if is_min_max_normalized(self.data.velocities):
            print("Velocity data is Min-Max Normalized")
        self.data.calculated_accelerations = min_max_scaling(self.data.calculated_accelerations)
        if is_min_max_normalized(self.data.calculated_accelerations):
            print("Acceleration data is Min-Max Normalized")

        self.test_data = list(zip(self.data.image_names, self.data.velocities, self.data.calculated_accelerations, 
                                  self.data.actions, self.data.actions_future_9, self.data.actions_future_12, self.data.actions_future_15,
                                  self.data.actions_future_18, self.data.actions_future_21))
        self.num_test_samples = len(self.test_data)
        
        print('Test samples: {0}'.format(self.num_test_samples))

    
   ###########################################################################
    #
    def _savefigs(self, plt, filename):
        plt.savefig(filename + '.png', dpi=150)
        plt.savefig(filename + '.pdf', dpi=150)
        print('Saved ' + filename + '.png & .pdf.')


    ###########################################################################
    #
    def _plot_results(self):
        # plt.figure()
        # # Plot a histogram of the prediction errors
        # num_bins = 25
        # hist, bins = np.histogram(self.steering_differences, num_bins)
        # center = (bins[:-1]+ bins[1:]) * 0.5
        # plt.bar(center, hist, width=0.05)
        # #plt.title('Historgram of Predicted Errors')
        # # if config['accel']:


         # all three histograms on one figure
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        num_bins = 25
        # Plot steering angle histogram with transparent bars
        hist, bins = np.histogram(self.steering_differences, num_bins)
        center = (bins[:-1] + bins[1:]) * 0.5
        ax1.bar(center, hist, width=0.05, alpha=0.5, color='b', label='Steering')
        # Plot throttle histogram with transparent bars
        hist, bins = np.histogram(self.throttle_differences, num_bins)
        center = (bins[:-1] + bins[1:]) * 0.5
        ax1.bar(center, hist, width=0.05, alpha=0.5, color='g', label='Throttle')
        # Plot brake histogram with transparent bars
        hist, bins = np.histogram(self.brake_differences, num_bins)
        center = (bins[:-1] + bins[1:]) * 0.5
        ax1.bar(center, hist, width=0.05, alpha=0.5, color='r', label='Brake')
        # title and labels
        ax1.set_title("Histograms of Prediction Errors")
        ax1.set_xlabel('Values')
        ax1.set_ylabel('Samples')
        ax1.legend()
        ax1.grid(True)
        # plt.show()
        self._savefigs(plt, self.data_path + '_str_thr_brk_err_hist')



        #######################################################################################
        # STEERING
        #######################################################################################

        # plt.xlabel('Steering Angle')
        # plt.ylabel('Number of Predictions')
        # plt.xlim(-1.0, 1.0)
        # plt.plot(np.min(self.differences), np.max(self.differences))
        # plt.tight_layout()
        # self._savefigs(plt, self.data_path + '_err_hist')

        plt.figure()
        # Plot a Scatter Plot of the Error
        plt.scatter(self.steering0_measurements, self.steering0_predictions)
        #plt.title('Scatter Plot of Errors')
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.axis('equal')
        plt.axis('square')
        plt.xlim([-1.0, 1.0])
        plt.ylim([-1.0, 1.0])
        plt.plot([-1.0, 1.0], [-1.0, 1.0], color='k', linestyle='-', linewidth=.1)
        plt.tight_layout()
        self._savefigs(plt, self.data_path + '_steering0_scatter')

        plt.figure()
        # Plot a Side-By-Side Comparison
        plt.plot(self.steering_measurements)
        plt.plot(self.steering_predictions)
        mean = sum(self.steering_differences)/len(self.steering_differences)
        variance = sum([((x - mean) ** 2) for x in self.steering_differences]) / len(self.steering_differences) 
        std = variance ** 0.5
        # print("mean:", mean)
        # print("std:", std)
        # plt.title('MAE: {0:.3f}, STDEV: {1:.3f}'.format(mean, std))
        #plt.title('Ground Truth vs. Prediction')
        plt.ylim([-1.0, 1.0])
        plt.xlabel('Time Step')
        plt.ylabel('Steering Angle')
        plt.legend(['ground truth', 'prediction'], loc='upper right')
        plt.tight_layout()
        self._savefigs(plt, self.data_path + '_steering_comparison')

        #######################################################################################
        # THROTTLE
        #######################################################################################

        # plt.xlabel('Throttle')
        # plt.ylabel('Number of Predictions')
        # plt.xlim(0.0, 1.0)
        # plt.plot(np.min(self.throttle_differences), np.max(self.throttle_differences))
        # plt.tight_layout()
        # self._savefigs(plt, self.data_path + '_thr_err_hist')

        plt.figure()
        # Plot a Scatter Plot of the Error
        plt.scatter(self.throttle_measurements, self.throttle_predictions)
        plt.title('Scatter Plot of Throttle Errors')
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.axis('equal')
        plt.axis('square')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.plot([0.0, 1.0], [0.0, 1.0], color='k', linestyle='-', linewidth=.1)
        plt.tight_layout()
        self._savefigs(plt, self.data_path + '_thr_scatter')

        plt.figure()
        # Plot a Side-By-Side Comparison
        plt.plot(self.throttle_measurements)
        plt.plot(self.throttle_predictions)
        mean_thr = sum(self.throttle_differences)/len(self.throttle_differences)
        variance_thr = sum([((x - mean_thr) ** 2) for x in self.throttle_differences]) / len(self.throttle_differences) 
        std_thr = variance_thr ** 0.5
        # plt.title('thr_MAE: {0:.3f}, thr_STDEV: {1:.3f}'.format(mean_thr, std_thr))
        #plt.title('Ground Truth vs. Prediction')
        plt.ylim([0.0, 1.0])
        plt.xlabel('Time Step')
        plt.ylabel('Throttle')
        plt.legend(['ground truth', 'prediction'], loc='upper right')
        plt.tight_layout()
        self._savefigs(plt, self.data_path + '_thr_comparison')

        #######################################################################################
        # BRAKE
        #######################################################################################

        # plt.xlabel('Brake')
        # plt.ylabel('Number of Predictions')
        # plt.xlim(0.0, 1.0)
        # plt.plot(np.min(self.brake_differences), np.max(self.brake_differences))
        # plt.tight_layout()
        # self._savefigs(plt, self.data_path + '_brk_err_hist')

        plt.figure()
        # Plot a Scatter Plot of the Error
        plt.scatter(self.brake_measurements, self.brake_predictions)
        plt.title('Scatter Plot of Brake Errors')
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.axis('equal')
        plt.axis('square')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.plot([0.0, 1.0], [0.0, 1.0], color='k', linestyle='-', linewidth=.1)
        plt.tight_layout()
        self._savefigs(plt, self.data_path + '_brk_scatter')

        plt.figure()
        # Plot a Side-By-Side Comparison
        plt.plot(self.brake_measurements)
        plt.plot(self.brake_predictions)
        mean_brk = sum(self.brake_differences)/len(self.brake_differences)
        variance_brk = sum([((x - mean_brk) ** 2) for x in self.brake_differences]) / len(self.brake_differences) 
        std_brk = variance_brk ** 0.5
        # plt.title('brk_MAE: {0:.3f}, brk_STDEV: {1:.3f}'.format(mean_brk, std_brk))
        #plt.title('Ground Truth vs. Prediction')
        plt.ylim([0.0, 1.0])
        plt.xlabel('Time Step')
        plt.ylabel('Brake')
        plt.legend(['ground truth', 'prediction'], loc='upper right')
        plt.tight_layout()
        self._savefigs(plt, self.data_path + '_brk_comparison')

            
        # show all figures
        plt.show()

    def _plot_results_delay_mlp_all(self):
        # plt.figure()
        # # Plot a histogram of the prediction errors
        # num_bins = 25
        # hist, bins = np.histogram(self.steering_differences, num_bins)
        # center = (bins[:-1]+ bins[1:]) * 0.5
        # plt.bar(center, hist, width=0.05)
        # #plt.title('Historgram of Predicted Errors')
        # # if config['accel']:


        # all three histograms on one figure
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        num_bins = 25
        # Plot steering angle histogram with transparent bars
        hist, bins = np.histogram(self.steering0_differences, num_bins)
        center = (bins[:-1] + bins[1:]) * 0.5
        ax1.bar(center, hist, width=0.05, alpha=0.5, label='Steering0')
        # Plot throttle histogram with transparent bars
        hist, bins = np.histogram(self.steering9_differences, num_bins)
        center = (bins[:-1] + bins[1:]) * 0.5
        ax1.bar(center, hist, width=0.05, alpha=0.5, label='Steering9')
        # Plot brake histogram with transparent bars
        hist, bins = np.histogram(self.steering12_differences, num_bins)
        center = (bins[:-1] + bins[1:]) * 0.5
        ax1.bar(center, hist, width=0.05, alpha=0.5, label='Steering12')
        hist, bins = np.histogram(self.steering15_differences, num_bins)
        center = (bins[:-1] + bins[1:]) * 0.5
        ax1.bar(center, hist, width=0.05, alpha=0.5, label='Steering15')
        hist, bins = np.histogram(self.steering18_differences, num_bins)
        center = (bins[:-1] + bins[1:]) * 0.5
        ax1.bar(center, hist, width=0.05, alpha=0.5, label='Steering18')
        hist, bins = np.histogram(self.steering21_differences, num_bins)
        center = (bins[:-1] + bins[1:]) * 0.5
        ax1.bar(center, hist, width=0.05, alpha=0.5, label='Steering21')
        # title and labels
        ax1.set_title("Histograms of Steering Prediction Errors")
        ax1.set_xlabel('Values')
        ax1.set_ylabel('Samples')
        ax1.legend()
        ax1.grid(True)
        # plt.show()
        self._savefigs(plt, self.data_path + '_str_delta_err_hist')



        #######################################################################################
        # STEERING delta 0
        #######################################################################################

        # plt.xlabel('Steering Angle')
        # plt.ylabel('Number of Predictions')
        # plt.xlim(-1.0, 1.0)
        # plt.plot(np.min(self.differences), np.max(self.differences))
        # plt.tight_layout()
        # self._savefigs(plt, self.data_path + '_err_hist')

        plt.figure()
        # Plot a Scatter Plot of the Error
        plt.scatter(self.steering0_measurements, self.steering0_predictions)
        plt.title('Steering Delta 0')
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.axis('equal')
        plt.axis('square')
        plt.xlim([-1.0, 1.0])
        plt.ylim([-1.0, 1.0])
        plt.plot([-1.0, 1.0], [-1.0, 1.0], color='k', linestyle='-', linewidth=.1)
        plt.tight_layout()
        self._savefigs(plt, self.data_path + '_steering0_scatter')

        plt.figure()
        # Plot a Side-By-Side Comparison
        plt.plot(self.steering0_measurements)
        plt.plot(self.steering0_predictions)
        mean = sum(self.steering0_differences)/len(self.steering0_differences)
        variance = sum([((x - mean) ** 2) for x in self.steering0_differences]) / len(self.steering0_differences) 
        std = variance ** 0.5
        # print("mean:", mean)
        # print("std:", std)
        # plt.title('MAE: {0:.3f}, STDEV: {1:.3f}'.format(mean, std))
        #plt.title('Ground Truth vs. Prediction')
        plt.ylim([-1.0, 1.0])
        plt.title('Steering Delta 0')
        plt.xlabel('Time Step')
        plt.ylabel('Steering Angle')
        plt.legend(['ground truth', 'prediction'], loc='upper right')
        plt.tight_layout()
        self._savefigs(plt, self.data_path + '_steering0_comparison')

        #######################################################################################
        # Steering delta 9
        #######################################################################################

        plt.figure()
        # Plot a Scatter Plot of the Error
        plt.scatter(self.steering9_measurements, self.steering9_predictions)
        plt.title('Steering Delta 9')
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.axis('equal')
        plt.axis('square')
        plt.xlim([-1.0, 1.0])
        plt.ylim([-1.0, 1.0])
        plt.plot([-1.0, 1.0], [-1.0, 1.0], color='k', linestyle='-', linewidth=.1)
        plt.tight_layout()
        self._savefigs(plt, self.data_path + '_steering9_scatter')

        plt.figure()
        # Plot a Side-By-Side Comparison
        plt.plot(self.steering9_measurements)
        plt.plot(self.steering9_predictions)
        mean = sum(self.steering9_differences)/len(self.steering9_differences)
        variance = sum([((x - mean) ** 2) for x in self.steering9_differences]) / len(self.steering9_differences) 
        std = variance ** 0.5
        # print("mean:", mean)
        # print("std:", std)
        # plt.title('MAE: {0:.3f}, STDEV: {1:.3f}'.format(mean, std))
        #plt.title('Ground Truth vs. Prediction')
        plt.ylim([-1.0, 1.0])
        plt.title('Steering Delta 9')
        plt.xlabel('Time Step')
        plt.ylabel('Steering Angle')
        plt.legend(['ground truth', 'prediction'], loc='upper right')
        plt.tight_layout()
        self._savefigs(plt, self.data_path + '_steering9_comparison')

        #######################################################################################
        # Steering delta 12
        #######################################################################################

        plt.figure()
        # Plot a Scatter Plot of the Error
        plt.scatter(self.steering12_measurements, self.steering12_predictions)
        plt.title('Steering Delta 12')
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.axis('equal')
        plt.axis('square')
        plt.xlim([-1.0, 1.0])
        plt.ylim([-1.0, 1.0])
        plt.plot([-1.0, 1.0], [-1.0, 1.0], color='k', linestyle='-', linewidth=.1)
        plt.tight_layout()
        self._savefigs(plt, self.data_path + '_steering12_scatter')

        plt.figure()
        # Plot a Side-By-Side Comparison
        plt.plot(self.steering12_measurements)
        plt.plot(self.steering12_predictions)
        mean = sum(self.steering12_differences)/len(self.steering12_differences)
        variance = sum([((x - mean) ** 2) for x in self.steering12_differences]) / len(self.steering12_differences) 
        std = variance ** 0.5
        # print("mean:", mean)
        # print("std:", std)
        # plt.title('MAE: {0:.3f}, STDEV: {1:.3f}'.format(mean, std))
        #plt.title('Ground Truth vs. Prediction')
        plt.ylim([-1.0, 1.0])
        plt.title('Steering Delta 12')
        plt.xlabel('Time Step')
        plt.ylabel('Steering Angle')
        plt.legend(['ground truth', 'prediction'], loc='upper right')
        plt.tight_layout()
        self._savefigs(plt, self.data_path + '_steering12_comparison')

        #######################################################################################
        # Steering delta 15
        #######################################################################################

        plt.figure()
        # Plot a Scatter Plot of the Error
        plt.scatter(self.steering15_measurements, self.steering15_predictions)
        plt.title('Steering Delta 15')
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.axis('equal')
        plt.axis('square')
        plt.xlim([-1.0, 1.0])
        plt.ylim([-1.0, 1.0])
        plt.plot([-1.0, 1.0], [-1.0, 1.0], color='k', linestyle='-', linewidth=.1)
        plt.tight_layout()
        self._savefigs(plt, self.data_path + '_steering15_scatter')

        plt.figure()
        # Plot a Side-By-Side Comparison
        plt.plot(self.steering15_measurements)
        plt.plot(self.steering15_predictions)
        mean = sum(self.steering15_differences)/len(self.steering15_differences)
        variance = sum([((x - mean) ** 2) for x in self.steering15_differences]) / len(self.steering15_differences) 
        std = variance ** 0.5
        # print("mean:", mean)
        # print("std:", std)
        # plt.title('MAE: {0:.3f}, STDEV: {1:.3f}'.format(mean, std))
        #plt.title('Ground Truth vs. Prediction')
        plt.ylim([-1.0, 1.0])
        plt.title('Steering Delta 15')
        plt.xlabel('Time Step')
        plt.ylabel('Steering Angle')
        plt.legend(['ground truth', 'prediction'], loc='upper right')
        plt.tight_layout()
        self._savefigs(plt, self.data_path + '_steering15_comparison')

        #######################################################################################
        # Steering delta 18
        #######################################################################################

        plt.figure()
        # Plot a Scatter Plot of the Error
        plt.scatter(self.steering18_measurements, self.steering18_predictions)
        plt.title('Steering Delta 18')
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.axis('equal')
        plt.axis('square')
        plt.xlim([-1.0, 1.0])
        plt.ylim([-1.0, 1.0])
        plt.plot([-1.0, 1.0], [-1.0, 1.0], color='k', linestyle='-', linewidth=.1)
        plt.tight_layout()
        self._savefigs(plt, self.data_path + '_steering18_scatter')

        plt.figure()
        # Plot a Side-By-Side Comparison
        plt.plot(self.steering18_measurements)
        plt.plot(self.steering18_predictions)
        mean = sum(self.steering18_differences)/len(self.steering18_differences)
        variance = sum([((x - mean) ** 2) for x in self.steering18_differences]) / len(self.steering18_differences) 
        std = variance ** 0.5
        # print("mean:", mean)
        # print("std:", std)
        # plt.title('MAE: {0:.3f}, STDEV: {1:.3f}'.format(mean, std))
        #plt.title('Ground Truth vs. Prediction')
        plt.ylim([-1.0, 1.0])
        plt.title('Steering Delta 18')
        plt.xlabel('Time Step')
        plt.ylabel('Steering Angle')
        plt.legend(['ground truth', 'prediction'], loc='upper right')
        plt.tight_layout()
        self._savefigs(plt, self.data_path + '_steering15_comparison')

        #######################################################################################
        # Steering delta 21
        #######################################################################################

        plt.figure()
        # Plot a Scatter Plot of the Error
        plt.scatter(self.steering21_measurements, self.steering21_predictions)
        plt.title('Steering Delta 21')
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.axis('equal')
        plt.axis('square')
        plt.xlim([-1.0, 1.0])
        plt.ylim([-1.0, 1.0])
        plt.plot([-1.0, 1.0], [-1.0, 1.0], color='k', linestyle='-', linewidth=.1)
        plt.tight_layout()
        self._savefigs(plt, self.data_path + '_steering21_scatter')

        plt.figure()
        # Plot a Side-By-Side Comparison
        plt.plot(self.steering21_measurements)
        plt.plot(self.steering21_predictions)
        mean = sum(self.steering21_differences)/len(self.steering21_differences)
        variance = sum([((x - mean) ** 2) for x in self.steering21_differences]) / len(self.steering21_differences) 
        std = variance ** 0.5
        # print("mean:", mean)
        # print("std:", std)
        # plt.title('MAE: {0:.3f}, STDEV: {1:.3f}'.format(mean, std))
        #plt.title('Ground Truth vs. Prediction')
        plt.ylim([-1.0, 1.0])
        plt.title('Steering Delta 15')
        plt.xlabel('Time Step')
        plt.ylabel('Steering Angle')
        plt.legend(['ground truth', 'prediction'], loc='upper right')
        plt.tight_layout()
        self._savefigs(plt, self.data_path + '_steering21_comparison')
            
        # show all figures
        plt.show()


   ###########################################################################
    #
    def run(self):
        
        self._prepare_data()
        #fname = self.data_path + const.LOG_EXT
        fname = self.data_path + const.LOG_EXT # use model name to save log
        
        file = open(fname, 'w')

        #print('image_name', 'label', 'predict', 'abs_error')
        bar = ProgressBar()

        # log file header
        if config['num_inputs'] == 3:
            if config['mlp_all']:
                file.write('image_name, velocity, acceleration, \
                    label_steering_angle_0, predicted_steering_angle_0, abs_steering0_error, squared_steering0_error,\
                    label_steering_angle_9, predicted_steering_angle_9, abs_steering9_error, squared_steering9_error,\
                    label_steering_angle_12, predicted_steering_angle_12, abs_steering12_error, squared_steering12_error,\
                    label_steering_angle_15, predicted_steering_angle_15, abs_steering15_error, squared_steering15_error,\
                    label_steering_angle_18, predicted_steering_angle_18, abs_steering18_error, squared_steering18_error,\
                    label_steering_angle_21, predicted_steering_angle_21, abs_steering21_error, squared_steering21_error\n')
            else:
                file.write('image_name, velocity, acceleration, \
                        label_steering_angle, predicted_steering_angle, abs_steering_error, squared_steering_error,\
                        label_throttle, predicted_throttle, abs_throttle_error, squared_throttle_error,\
                        label_brake, predicted_brake, abs_brake_error, squared_brake_error \n')
        elif config['num_inputs'] == 2:
            file.write('image_name, velocity, \
                    label_steering_angle, predicted_steering_angle, abs_steering_error, squared_steering_error,\
                    label_throttle, predicted_throttle, abs_throttle_error, squared_throttle_error,\
                    label_brake, predicted_brake, abs_brake_error, squared_brake_error \n')

        
        # if Config.neural_net['str_out']:
        #     file.write('image_name, velocity, throttle, brake, label_steering_angle, pred_steering_angle, abs_error, squared_error\n')
        # elif Config.neural_net['thr_out']:
        #     file.write('image_name, velocity, steering, brake, label_throttle, pred_throttle, abs_error, squared_error\n')
        # elif Config.neural_net['brk_out']:
        #     file.write('image_name, velocity, steering, throttle, label_brake, pred_brake, abs_error, squared_error\n')
        # elif Config.neural_net['accel']:
        #     file.write('image_name, velocity, acceleration, label_throttle, label_brake, \
        #                pred_throttle, pred_brake, thr_abs_error, thr_squared_error, brk_abs_error, brk_squared_error\n')

        if config['lstm'] is True:
            images = []
            #images_names = []
            cnt = 1

            for image_name, velocity, measurement in bar(self.test_data):   
                image_fname = self.data_path + '/' + image_name
                image = cv2.imread(image_fname)

                # if collected data is not cropped then crop here
                # otherwise do not crop.
                if config_dc['crop'] is not True:
                    image = image[config_dc['image_crop_y1']:config_dc['image_crop_y2'],
                                config_dc['image_crop_x1']:config_dc['image_crop_x2']]

                image = cv2.resize(image, (config['input_image_width'],
                                        config['input_image_height']))
                image = self.image_process.process(image)

                images.append(image)
                #images_names.append(image_name)
                
                if cnt >= config['lstm_timestep']:
                    trans_image = np.array(images).reshape(-1, config['lstm_timestep'], 
                                                config['input_image_height'],
                                                config['input_image_width'],
                                                config['input_image_depth'])                    

                    predict = self.net_model.model.predict(trans_image)
                    pred_steering_angle = predict[0][0]
                    pred_steering_angle = pred_steering_angle / config['steering_angle_scale']
                
                    if config['num_outputs'] == 2:
                        pred_throttle = predict[0][1]
                    
                    label_steering_angle = measurement[0] # labeled steering angle
                    self.measurements.append(label_steering_angle)
                    self.predictions.append(pred_steering_angle)
                    diff = abs(label_steering_angle - pred_steering_angle)
                    self.differences.append(diff)
                    self.squared_differences.append(diff*2)
                    log = image_name+','+str(label_steering_angle)+','+str(pred_steering_angle)\
                                    +','+str(diff)\
                                    +','+str(diff**2)

                    file.write(log+'\n')
                    # delete the 1st element
                    del images[0]
        else:
            for image_name, velocity, acceleration, action, action_9, action_12, action_15, action_18, action_21 in bar(self.test_data):   
                image_fname = self.data_path + '/' + image_name
                image = cv2.imread(image_fname)

                # if collected data is not cropped then crop here
                # otherwise do not crop.
                if config_dc['crop'] is not True:
                    image = image[config_dc['image_crop_y1']:config_dc['image_crop_y2'],
                                config_dc['image_crop_x1']:config_dc['image_crop_x2']]

                image = cv2.resize(image, (config['input_image_width'],
                                        config['input_image_height']))
                image = self.image_process.process(image)
                
                npimg = np.expand_dims(image, axis=0)
                np_velocity = np.array(velocity).reshape(-1, 1)
                np_acceleration = np.array(acceleration).reshape(-1,1)

                steering_angle, throttle, brake, throttle_brake = action 
                steering_angle_9, throttle_9, brake_9, throttle_brake_9 = action_9
                steering_angle_12, throttle_12, brake_12, throttle_brake_12 = action_12
                steering_angle_15, throttle_15, brake_15, throttle_brake_15 = action_15
                steering_angle_18, throttle_18, brake_18, throttle_brake_18 = action_18
                steering_angle_21, throttle_21, brake_21, throttle_brake_21 = action_21
                if config['delay_mitig']:
                    if config['mlp_all']:
                        steer_true = [steering_angle, steering_angle_9, steering_angle_12, 
                                      steering_angle_15, steering_angle_18, steering_angle_21]
                    else:
                        if config['delta'] == 9:
                            steering_angle, throttle, brake = steering_angle_9, throttle_9, brake_9
                        elif config['delta'] == 12:
                            steering_angle, throttle, brake = steering_angle_12, throttle_12, brake_12
                        elif config['delta'] == 15:
                            steering_angle, throttle, brake = steering_angle_15, throttle_15, brake_15
                        elif config['delta'] == 18:
                            steering_angle, throttle, brake = steering_angle_18, throttle_18, brake_18
                        elif config['delta'] == 21:
                            steering_angle, throttle, brake = steering_angle_21, throttle_21, brake_21
                    
                # print('steering type')
                # print(type(steering_angle))
                
                # np_steering_angle = np.array(steering_angle).reshape(-1, 1)
                np_throttle = np.array(throttle).reshape(-1, 1)
                np_brake = np.array(brake).reshape(-1, 1)

                if config['num_inputs'] == 3:
                    predict = self.net_model.model.predict([npimg, np_velocity, np_acceleration])
                    if config['mlp_all']:
                        predict_base = self.net_model.base_model.predict([npimg, np_velocity, np_acceleration])
                elif config['num_inputs'] == 2:
                    predict = self.net_model.model.predict([npimg, np_velocity])
                else:
                    predict = self.net_model.model.predict(npimg)

                if not config['mlp_all']:
                    pred_steering_angle = predict[0][0]
                    # print('pred_steering_angle type')
                    # print(type(pred_steering_angle))
                    if config['num_outputs'] == 3:
                        pred_throttle_brake = predict[1][0]

                    pred_throttle=[]
                    pred_brake=[]
                    for i in range(len(pred_throttle_brake)):
                        if pred_throttle_brake[i] >=0.0:
                            pred_throttle.append(pred_throttle_brake[i])
                            pred_brake.append(0.0)
                        else:
                            pred_throttle.append(0.0)
                            pred_brake.append(-1*pred_throttle_brake[i])
                    # Convert lists to numpy arrays for element-wise operations
                    pred_throttle = np.array(pred_throttle)
                    pred_brake = np.array(pred_brake)

                    # label_steering_angle = (steering_angle+1.0)/2.0    
                    label_steering_angle = np.array(steering_angle)
                    label_throttle = throttle
                    label_brake = brake
                    
                    self.steering_measurements.append(label_steering_angle)
                    self.steering_predictions.append(pred_steering_angle)
                    diff_steering = abs(label_steering_angle - pred_steering_angle)
                    self.steering_differences.append(diff_steering[0])
                    # print(self.steering_differences)
                    self.steering_squared_differences.append(diff_steering**2)
                    
                    self.throttle_measurements.append(label_throttle)
                    self.throttle_predictions.append(pred_throttle)
                    diff_thr = abs(label_throttle - pred_throttle)
                    self.throttle_differences.append(diff_thr)
                    self.throttle_squared_differences.append(diff_thr**2)
                    
                    self.brake_measurements.append(label_brake)
                    self.brake_predictions.append(pred_brake)
                    diff_brk = abs(label_brake - pred_brake)
                    self.brake_differences.append(diff_brk)
                    self.brake_squared_differences.append(diff_brk**2)

                    # file.write('image_name, velocity, acceleration, \
                    #     label_steering_angle, predicted_steering_angle, abs_steering_error, squared_steering_error,\
                    #     label_throttle, predicted_throttle, abs_throttle_error, squared_throttle_error,\
                    #     label_brake, predicted_brake, abs_brake_error, squared_brake_error \n')
                                                    
                    if config['num_inputs'] == 3:
                        log = image_name+','+ str(velocity) +','+ str(acceleration) + ',' \
                                + str(label_steering_angle) + ',' + str(pred_steering_angle) +',' + str(diff_steering) + ',' + str(diff_steering**2) + ',' \
                                + str(label_throttle) + ',' + str(pred_throttle) +',' + str(diff_thr) + ',' + str(diff_thr**2) + ',' \
                                + str(label_brake) + ',' + str(pred_brake) +',' + str(diff_brk) + ',' + str(diff_brk**2) 
                    elif config['num_inputs'] == 2:
                        log = image_name+','+ str(velocity) +',' \
                                + str(label_steering_angle) + ',' + str(pred_steering_angle) +',' + str(diff_steering) + ',' + str(diff_steering**2) + ',' \
                                + str(label_throttle) + ',' + str(pred_throttle) +',' + str(diff_thr) + ',' + str(diff_thr**2) + ',' \
                                + str(label_brake) + ',' + str(pred_brake) +',' + str(diff_brk) + ',' + str(diff_brk**2) 
                    file.write(log+'\n')
                else:
                    pred_steering_angle_0 = predict_base[0][0]
                    # print('pred_steering_angle type')
                    # print(type(pred_steering_angle))
                    pred_steering_angle_9 = predict[0][0]
                    pred_steering_angle_12 = predict[1][0]
                    pred_steering_angle_15 = predict[2][0]
                    pred_steering_angle_18 = predict[3][0]
                    pred_steering_angle_21 = predict[4][0]


                    # label_steering_angle = (steering_angle+1.0)/2.0    
                    label_steering_angle_0 = np.array(steer_true[0])
                    label_steering_angle_9 = np.array(steer_true[1])
                    label_steering_angle_12 = np.array(steer_true[2])
                    label_steering_angle_15 = np.array(steer_true[3])
                    label_steering_angle_18 = np.array(steer_true[4])
                    label_steering_angle_21 = np.array(steer_true[5])


                    
                    self.steering0_measurements.append(label_steering_angle_0)
                    self.steering0_predictions.append(pred_steering_angle_0)
                    diff_steering0 = abs(label_steering_angle_0 - pred_steering_angle_0)
                    self.steering0_differences.append(diff_steering0[0])
                    # print(self.steering_differences)
                    self.steering0_squared_differences.append(diff_steering0**2)

                    self.steering9_measurements.append(label_steering_angle_9)
                    self.steering9_predictions.append(pred_steering_angle_9)
                    diff_steering9 = abs(label_steering_angle_9 - pred_steering_angle_9)
                    self.steering9_differences.append(diff_steering9[0])
                    self.steering9_squared_differences.append(diff_steering9**2)
                    
                    self.steering12_measurements.append(label_steering_angle_12)
                    self.steering12_predictions.append(pred_steering_angle_12)
                    diff_steering12 = abs(label_steering_angle_12 - pred_steering_angle_12)
                    self.steering12_differences.append(diff_steering12[0])
                    self.steering12_squared_differences.append(diff_steering12**2)

                    self.steering15_measurements.append(label_steering_angle_15)
                    self.steering15_predictions.append(pred_steering_angle_15)
                    diff_steering15 = abs(label_steering_angle_15 - pred_steering_angle_15)
                    self.steering15_differences.append(diff_steering15[0])
                    self.steering15_squared_differences.append(diff_steering15**2)

                    self.steering18_measurements.append(label_steering_angle_18)
                    self.steering18_predictions.append(pred_steering_angle_18)
                    diff_steering18 = abs(label_steering_angle_18 - pred_steering_angle_18)
                    self.steering18_differences.append(diff_steering18[0])
                    self.steering18_squared_differences.append(diff_steering18**2)

                    self.steering21_measurements.append(label_steering_angle_21)
                    self.steering21_predictions.append(pred_steering_angle_21)
                    diff_steering21 = abs(label_steering_angle_21 - pred_steering_angle_21)
                    self.steering21_differences.append(diff_steering21[0])
                    self.steering21_squared_differences.append(diff_steering21**2)

                    # file.write('image_name, velocity, acceleration, \
                    #     label_steering_angle, predicted_steering_angle, abs_steering_error, squared_steering_error,\
                    #     label_throttle, predicted_throttle, abs_throttle_error, squared_throttle_error,\
                    #     label_brake, predicted_brake, abs_brake_error, squared_brake_error \n')
                                                    
                    if config['num_inputs'] == 3:
                        log = image_name+','+ str(velocity) +','+ str(acceleration) + ',' \
                                + str(label_steering_angle_0) + ',' + str(pred_steering_angle_0) +',' + str(diff_steering0) + ',' + str(diff_steering0**2) + ',' \
                                + str(label_steering_angle_9) + ',' + str(pred_steering_angle_9) +',' + str(diff_steering9) + ',' + str(diff_steering9**2) + ',' \
                                + str(label_steering_angle_12) + ',' + str(pred_steering_angle_12) +',' + str(diff_steering12) + ',' + str(diff_steering12**2) + ',' \
                                + str(label_steering_angle_15) + ',' + str(pred_steering_angle_15) +',' + str(diff_steering15) + ',' + str(diff_steering15**2) + ',' \
                                + str(label_steering_angle_18) + ',' + str(pred_steering_angle_18) +',' + str(diff_steering18) + ',' + str(diff_steering18**2) + ',' \
                                + str(label_steering_angle_21) + ',' + str(pred_steering_angle_21) +',' + str(diff_steering21) + ',' + str(diff_steering21**2) 
                    elif config['num_inputs'] == 2:
                        log = image_name+','+ str(velocity) +',' \
                                + str(label_steering_angle) + ',' + str(pred_steering_angle) +',' + str(diff_steering) + ',' + str(diff_steering**2) + ',' \
                                + str(label_throttle) + ',' + str(pred_throttle) +',' + str(diff_thr) + ',' + str(diff_thr**2) + ',' \
                                + str(label_brake) + ',' + str(pred_brake) +',' + str(diff_brk) + ',' + str(diff_brk**2) 
                    file.write(log+'\n')
                
        file.close()
        print('Saved ' + fname + '.')

        if config['mlp_all']:
            self._plot_results_delay_mlp_all()
        else:
            self._plot_results()



# from drive_log_acc import DriveLog


###############################################################################
#       
def main(base_model_path, model_path, data_folder_name):
    drive_log = DriveLog(base_model_path, model_path, data_folder_name) 
    drive_log.run() # data folder path to test
       

###############################################################################
#       
if __name__ == '__main__':
    import sys

    try:
        if (len(sys.argv) != 4):
            exit('Usage:\n$ python {} base_model_path model_path data_folder_name'.format(sys.argv[0]))
        
        main(sys.argv[1], sys.argv[2], sys.argv[3]) # base_model_path, model_path, data_folder_name

    except KeyboardInterrupt:
        print ('\nShutdown requested. Exiting...')
