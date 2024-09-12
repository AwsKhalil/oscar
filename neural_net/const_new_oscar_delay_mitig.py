#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 13:23:14 2017
History:
11/28/2020: modified for OSCAR 

@author: jaerock
"""
###############################################################################
# constant definition

# network model type
NET_TYPE_JAEROCK                                        = 0
NET_TYPE_CE491                                          = 1
NET_TYPE_JAEROCK_VEL                                    = 2
NET_TYPE_JAEROCK_SHIFT                                  = 3 
NET_TYPE_STR_THR_BRK                                    = 4
NET_TYPE_LSTM_DELAY_MITIGATION                          = 5
NET_TYPE_CONDITIONAL_MLP_DELAY_MITIGATION               = 6
NET_TYPE_CONDITIONAL_MLP_DELAY_MITIGATION_ALL           = 7
NET_TYPE_CONDITIONAL_MLP_DELAY_MITIGATION_ALL_2         = 8
NET_TYPE_CONDITIONAL_LSTM_MLP_DELAY_MITIGATION_ALL      = 9
NET_TYPE_CONDITIONAL_LSTM_MLP_DELAY_MITIGATION_ALL_2    = 10


NET_TYPE_CONVLSTM             = 11
NET_TYPE_CONVLSTM_IMG_OUT     = 12

NET_TYPE_ROVER_MULTICLASS_STR = 13



# file extension
DATA_EXT             = '.csv'
IMAGE_EXT            = '.jpg'
LOG_EXT              = '_log.csv'