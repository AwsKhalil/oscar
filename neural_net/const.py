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
# NET_TYPE_JAEROCK      = 0
NET_TYPE_JAEROCK_ELU  = 0
NET_TYPE_JAEROCK_ELU_360  = 10
NET_TYPE_CE491        = 1
NET_TYPE_JAEROCK_VEL  = 2

NET_TYPE_SAP    = 31
NET_TYPE_VGG16  = 32
NET_TYPE_RESNET = 33
NET_TYPE_ALEXNET= 34

NET_TYPE_DONGHYUN = 11
NET_TYPE_DONGHYUN2= 12
NET_TYPE_DONGHYUN3= 13
NET_TYPE_DONGHYUN4= 14
NET_TYPE_DONGHYUN5= 15
NET_TYPE_DONGHYUN6= 16

NET_TYPE_LRCN        = 21
NET_TYPE_SPTEMLSTM   = 22
NET_TYPE_COOPLRCN    = 23

# file extension
DATA_EXT             = '.csv'
IMAGE_EXT            = '.jpg'
LOG_EXT              = '_log.csv'