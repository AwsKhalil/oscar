# OSCAR

|     |     |
| --- | --- |
|![](imgs4readme/Screenshot%20from%202023-02-02%2015-50-55.png) |![](imgs4readme/Screenshot%20from%202023-02-02%2015-51-22.png) |
|![](imgs4readme/Screenshot%20from%202023-02-02%2015-48-22.png)|![](imgs4readme/Screenshot%20from%202023-02-02%2015-48-37.png)|
|![](imgs4readme/Screenshot%20from%202023-02-02%2015-49-35.png) |![](imgs4readme/Screenshot%20from%202023-02-02%2015-49-47.png) |

## History
- 12/17/2023:
  
  - **Overview**:
    - New Gazebo plugin for the IMU sensor
    - New node (Acceleration) which calculates the acceleration based on velocity and time
    - Modified the data_colllection node to include IMU data, heading, and acceleration.
    - Modified the data_colllection node by adding time Synchronizing.
    - New driving model (input: [image, velocity, acceleration] --> output: [steering, throttle, brake])

  - **Detailed modifications**:
    - oscar/
      - start_fusion.sh
      - collect_data_fusion.sh 
    - oscar/catkin_ws/src
      - added the "acceleration" package
      - added data_collection/scripts/data_collection_sync3.py
      - fusion/robot_description/urdf/fusion.urdf
      - fusion/launch/sitl.launch
      - added 3 files to fusion/worlds
      - run_neural/scripts/run_neural.py
        
    - oscar/config:
      - config.yaml
      - added new config files
        - data_collection/fusion_aws.yaml
        - neural_net/fusion_template_aws.yaml
        - run_neural/fusion_template_aws.yaml
      
    - oscar/neural_net
      - const.py
      - data_augmentation.py
      - drive_data.py
      - drive_log.py
      - drive_run.py
      - drive_train.py
      - image_process.py
      - net_model.py
      - rebuild_csv.py
      - train.py
      

  
- 3/11/2021: 
  - Removed unused plugins that are not compatible with Gazebo-11.
  - Works with Gazebo-11 as well. 
  - Tested in Ubuntu 20.04 and ROS Noetic. Only checked basic function and not yet fully tested.
- 3/06/2021: Version 1.6 released
  - Input can have velocity in addition to image. See `config/neural_net [num_input]`. 
  - Output can have throttle in addtion to steering angle. `See config/neural_net [num_output]`.
  - Add `brake` in `config/data_collection` (config version # 0.92 was used previously).
- 2/18/2021: Version 1.5 released.
  - Add 'brake' to the `data_collection`.
  - Update the modules that use data from `data_collection`.
- 01/15/2021: Version 1.4 released.
  - Split one config (neural_net) to three (neural_net, data_collection, run_neural)
- 12/04/2020: Version 1.2 released.
  - New data collection policy.
  - Data normalization added.
- 11/30/2020: Version 1.0 released.

## Introduction

OSCAR is the Open-Source robotic Car Architecture for Research and education. OSCAR is an open-source and full-stack robotic car architecture to advance robotic research and education in a setting of autonomous vehicles.

The OSCAR platform was designed in the Bio-Inspired Machine Intelligence Lab at the University of Michigan-Dearborn. 

The OSCAR supports two vehicles: `fusion` and `rover`.

`fusion` is based on `car_demo` from OSRF that was originally developed to test simulated Toyota Prius energy efficiency.

The backend system of `rover` is the PX4 Autopilot with Robotic Operating System (ROS) communicating with PX4 running on hardware or on the Gazebo simulator. 

## Who is OSCAR for?

The OSCAR platform can be used by researchers who want to have a full-stack system for a robotic car that can be used in autonomous vehicles and mobile robotics research.
OSCAR helps educators who want to teach mobile robotics and/or autonomous vehicles in the classroom. 
Students also can use the OSCAR platform to learn the principles of robotics programming.

## Download the OSCAR Source Code

```
$ git clone https://github.com/jrkwon/oscar.git --recursive
```

## Directory Structure
- `catkin_ws`: ros workspace
  - `src`
    - `data_collection`: data from front camera and steering/throttle
    - `fusion`: Ford Fusion Energia model
    - `rover`: 
- `config`: configurations
  - `conda`: conda environment files
  - `config.yaml`: config file names for neural_net, data_collection, and run_neural
  - `neural_net`: system settings for neural_net
  - `data_collection`: system settings for data_collection
  - `run_neural`: system settings for run_neural
- `neural_net`: neural network package for end to end learning
- `PX4-Autopilot`: The folder for the PX4 Autopilot.

## Prior to Use

### Versions 

The OSCAR has been tested with ROS Melodic on Ubuntu 18.04.

### Install ROS packages
Install two more packages for this project unless you already have them in your system.
```
$ sudo apt install ros-$ROS_DISTRO-fake-localization
$ sudo apt install ros-$ROS_DISTRO-joy

```

### Create Conda Environment 

Create a conda environment using an environment file that is prepared at `config/conda`.
```
$ conda env create --file config/conda/environment.yaml
```
### rover only
This section applies to `rover` which is based on `PX4 `. When RC signal is lost, the vehicle's default behavior is `homing` with `disarming` to protect the vehicle. 
We disabled this feature to prevent the vehicle from disarming whenever control signals are not being sent.

Use QGroundControl to disable the feature. Find `COM_OBLACT` and make it `Disable`.

## How to Use

### Activate Conda Environment

Activate the `oscar` environment. 
```
$ conda activate oscar
```


This section explains how to use `fusion` and `rover`.

### fusion

`fusion` is heavily relied on OSRF's `car_demo` project. Simply use the following script.

```
(oscar) $ ./start_fusion.sh 
```

A `world` can be selected through a command line argument. Three worlds are ready to be used.
- `track_jaerock`: This is default. No need to specified.
- `sonoma_raceway`: racetrack
- `mcity_jaerock`: mcity
- `track_aws_ANEC`: training track 3 lanes instead of 2
- `track_aws_ANEC_test`: testing track 3 lanes instead of 2
- `track_aws_smc`: The start and end points are not the same in this track

```
(oscar) $ ./start_fusion.sh track_aws_ANEC
```

### rover 

`rover` is based on the Software-In-The-Loop of PX4.

1. Start the rover

```
(oscar) $ ./start_rover.sh
```

2. Get rover ready to be controlled.

Open a new terminal and run the following shell scripts.
```
(oscar) $ ./cmd_arming.sh
(oscar) $ ./offboard_mode.sh
```

Then the `rover` is ready to be controlled by the topic `/mavros/setpoint_velocity/cmd_vel` or `/mavros/setpoint_velocity/cmd_vel_unstamped`. The `OSCAR` uses the `unstamped` version.

## How to Collect Data

Run the script with a data ID as an argument.
```
(oscar) $ ./collect_data_fusion data1
```

The default data folder location is `$(pwd)e2e_{fusion/rover}_data`.

### Data Format

From `data_collection` config version 0.92, the CSV file has one more column for `brake`. Use `convert_csv.py` to convert a data CSV file collected before 0.92 to a new CSV file.

#### From Version 2023

Data Collection will save a csv file with images. The CSV file has the following columns

```
image_file_name / steering_angle / throttle / brake / linux_time / velocity / velocity_x / velocity_y / velocity_z / position_x / position_y / position_z/ imu_acceleration_x / imu_acceleration_z / imu_acceleration_z / yaw_rate / heading / calculated_acceleration / time_stamp

```

#### From Version 0.92

Data Collection will save a csv file with images. The CSV file has the following columns

```
image_file_name / steering_angle / throttle / brake / linux_time / velocity / velocity_x / velocity_y / velocity_z / position_x / position_y / position_z 

```

```
2020-12-08-23-55-31-150079.jpg	-0.0149267930537	0.15	0.7 1607489731.15	0.846993743317	0.846750728334	-0.00903874268025	-0.0181633261171	8.25840907119	-102.836707258	0.0248406100056

```

#### Before Version 0.92

Data Collection will save a csv file with images. The CSV file has the following columns

```
image_file_name / steering_angle / throttle / linux_time / velocity / velocity_x / velocity_y / velocity_z / position_x / position_y / position_z

```

```
2020-12-08-23-55-31-150079.jpg	-0.0149267930537	0.15	1607489731.15	0.846993743317	0.846750728334	-0.00903874268025	-0.0181633261171	8.25840907119	-102.836707258	0.0248406100056

```

## Data Cleaning

When some of test images must be deleted, just delete them and rebuild the csv using `rebuild_csv.py`.

```
(oscar) $ python rebuild_csv.py path/to/data/folder
```

## How to Train Neural Network

### steering_angle_scale

`steering_angle_scale` in `neural_net` config is for making the neural network have higher precision in prediction. The range of steering angle is -1 to 1. But in most cases, there will not be values between -1 and -0.5 as well as between 0.5 to 1 which means very sharp steering angles. These sharp steering angles will not be collected from driving a track in practice.

To find a proper scale value, you may use `test_data.py` by which you can see data statistics. The following is an example.

The choice of `steering_angle_scale` is especially important when activation functions are `sigmoid` or `tanh` in which you may lose data samples of sharp turns.

```
####### data statistics #########
Steering Command Statistics:
count    6261.000000
mean        0.002407
std         0.134601
min        -0.421035
25%        -0.016988
50%         0.009774
75%         0.085238
max         0.310105
Name: steering_angle, dtype: float64
```

### Training
Start a training
```
(oscar) $ . setup.bash
(oscar) $ python neural_net/train.py path/to/data/folder
```

### TensorBoard

After starting a training session, start tensorboard.
```
(oscar) $ tensorboard --logdir ./logs/scalars/
```

## How to Test Neural Network

TBA
```
(oscar) $ . setup.bash
(oscar) $ python neural_net/test.py path/to/data/model path/to/data/folder
```

To compare labeled steering angles and their corresponding ones.
```
(oscar) $ . setup.bash
(oscar) $ python neural_net/drive_log.py path/to/data/model path/to/data/folder
```

## How to See Saliency Map

TBA
```
(oscar) $ . setup.bash
(oscar) $ python neural_net/test_saliency.py path/to/data/model path/to/image/to/test
```

## How to Drive using Neural Network

TBA
```
(oscar) $ . setup.bash
(oscar) $ rosrun run_neural run_nerual.py path/to/data/model 
```

## How to See Collected Data with/without Inference

### Visualization of Steering Angle Predictions
You can specify a trained neural network model to see how the inference engine is actually working.
```
(oscar) $ python neural_net/drive_view.py path/to/data/model path/to/data/folder path/to/folder/to/save 
```

### Visualization of Collected Data
It is also possible to visualize collected data with other information without steering angle predictions.
```
(oscar) $ python neural_net/drive_view.py path/to/data/folder path/to/folder/to/save 
```

## Sample Datasets

The datasets below were collected before we added `brake`. The CSV files must be converted by `convert_csv.py` before being used.

### New sample datasets (not cropped)
- https://drive.google.com/drive/folders/197w7u99Jvyf5tuRTTawLhaYuwBLSxo-O?usp=sharing

### Legacy sample datasets (cropped)
- https://drive.google.com/drive/folders/173w5kh9h5QCDG8LEJPQ1qGJKwz1TnAkr?usp=sharing

## ROS packages 

```
$ sudo apt install ros-$ROS_DISTRO-hector-gazebo-plugins
```

## Relevant Publications

- Jaerock Kwon, Aws Khalil, Donghyun Kim, Haewoon Nam, “Incremental End-to-End Learning for Lateral Control in Autonomous Driving,” IEEE Access, 2022.
- Aws Khalil, Ahmed Abdelhamed, Girma Tewolde, Jaerock Kwon, “Ridon Vehicle: Drive-by-Wire System for Scaled Vehicle Platform and Its Application on Behavior Cloning,” MDPI Energies, Special Issue “Autonomous Vehicles Perception and Control,” 2021.
- Byung Chan Choi, Jaerock Kwon, Haewoon Nam, “Image Prediction for Lane Following Assist using Convolutional Neural Network-based U-Net,” The 4th International Conference on Artificial Intelligence in Information and Communication (ICAIIC 2022), Feb 21-24, 2022
- Donghyun Kim, Jaerock Kwon, Haewoon Nam, “End-to-End Learning-based Self-Driving Control Imitating Human Driving”, The 12th International Conference on ICT Convergence (ICTC 2021), Jeju Island, Korea, October 20-22, 2021
- Ahmed Abdelhamed, Girma Tewolde, and Jaerock Kwon, Simulation Framework for Development and Testing of Autonomous Vehicles, International IOT, Electronics and Mechatronics Conference, Vancouver, Canada, Sep 2020
- Nikhil Prabhu, Sewoong Min, Haewoon Nam, Girma Tewolde, and Jaerock Kwon, Integrated Framework of Autonomous Vehicle with Traffic Sign Recognition in Simulation Environment, IEEE International Conference on Electro/Information Technology, Naperville, IL, USA, July 2020
- Shobit Sharma, Girma Tewolde, and Jaerock Kwon, Lateral and Longitudinal Motion Control of Autonomous Vehicles using Deep Learning, IEEE International Conference on Electro/Information Technology, Brookings, South Dakota, USA, May 2019

## Acknowledgments

### System Design and Implementation

- Jaerock Kwon, Ph.D.: Assistant Professor of Electrical and Computer Engineering at the University of Michigan-Dearborn

### Implementation and Contributors

- Donghyun Kim: Ph.D. student at Hanyang University-ERICA, Korea
- Aws Khalil: Ph.D. student at the University of Michigan-Dearborn
- Jesudara Omidokun: M.S. student at the University of Michigan-Dearborn

### References

- https://github.com/osrf/car_demo
- https://github.com/ICSL-hanyang/uv_base/tree/Go-kart
- https://github.com/PX4/PX4-Autopilot
