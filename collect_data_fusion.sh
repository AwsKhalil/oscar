#!/bin/bash

source setup.bash

rosparam set path_to_e2e_data $(pwd)/e2e_fusion_data  # /path/to/data
# rosparam set /use_sim_time true
rosrun data_collection data_collection_delay_mitig.py $1 /camera/image_raw:=/fusion/front_camera/image_raw
# rosrun data_collection data_collection_sync3.py $1 /camera/image_raw:=/fusion/front_camera/image_raw
# rosrun data_collection data_collection_nosync_new.py $1 /camera/image_raw:=/fusion/front_camera/image_raw
