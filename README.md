# Bimur
Bimur (Bi-manual UR5). This repository is for packages related to UR5 at MuLIP Lab, Tufts. This branch was created by updating the master branch. 

Packages that were updated: 

## Universal_Robots_ROS_Driver
```
git clone https://github.com/UniversalRobots/Universal_Robots_ROS_Driver.git
```

## robotiq_85_gripper  
```
git clone https://github.com/GT-RAIL/robotiq_85_gripper.git
```

## universal_robots
```
git clone https://github.com/ros-industrial/universal_robot.git
```

Followed these steps from 
```
git clone -b calibration_devel https://github.com/fmauch/universal_robot.git 
```
# install dependencies
$ sudo apt update -qq
$ rosdep update
$ rosdep install --from-paths src --ignore-src -y


<img src="pics/Bimur.png" align="middle">

# Installation

`git clone https://github.com/tufts-ai-robotics-group/Bimur.git`

## Requirements

1. Ubuntu 18.04
2. ROS Melodic
3. Gazebo x.x


## ROS Packages
```
sudo apt-get install ros-melodic-ur-msgs
sudo apt-get install ros-melodic-ur-client-library 

```

## Other Packages
```
sudo apt-get install ros-melodic-catkin-virtualenv
sudo apt-get install ros-melodic-control-toolbox
sudo apt-get install ros-melodic-controller-interface
sudo apt-get install ros-melodic-controller-manager-msgs
sudo apt-get install ros-melodic-controller-manager
sudo apt-get install ros-melodic-effort-controllers
sudo apt-get install ros-melodic-force-torque-sensor-controller
sudo apt-get install ros-melodic-gazebo-ros-control
sudo apt-get install ros-melodic-industrial-robot-status-interface
sudo apt-get install ros-melodic-joint-state-controller
sudo apt-get install ros-melodic-joint-state-publisher-gui
sudo apt-get install ros-melodic-joint-trajectory-controller
sudo apt-get install ros-melodic-moveit
sudo apt-get install ros-melodic-moveit-core
sudo apt-get install ros-melodic-moveit-kinematics
sudo apt-get install ros-melodic-moveit-ros-visualization
sudo apt-get install ros-melodic-position-controllers
sudo apt-get install ros-melodic-rqt-joint-trajectory-controller
sudo apt-get install ros-melodic-socketcan-interface
sudo apt-get install ros-melodic-soem
sudo apt-get install ros-melodic-speech-recognition-msgs
sudo apt-get install ros-melodic-ur-client-library
sudo apt-get install ros-melodic-velocity-controllers

```




Following GitHub repo. was used (no need to clone them):
```
git clone https://github.com/UniversalRobots/Universal_Robots_ROS_Driver.git
git checkout b466518

git clone -b calibration_devel https://github.com/fmauch/universal_robot.git
git checkout e5a176c

git clone https://github.com/StanleyInnovation/robotiq_85_gripper
git checkout 2240a8c

git clone https://github.com/ros-industrial/robotiq.git
git checkout 66961ec
```

# Right Robot

## Simulation

### Rviz
Launch URDF in Rviz for visualization <br>
`roslaunch bimur_description bimur_right_robot_rviz.launch`

### Gazebo
Launch URDF in Gazebo with specific joint poses: <br>
`roslaunch bimur_bringup bimur_right_robot_gazebo_1.launch`

Launch URDF in Gazebo with specific kinematics_config: <br>
`roslaunch bimur_bringup bimur_right_robot_gazebo_2.launch`

### Gazebo + MoveIt Rviz:
UR5 Arm: <br>
`roslaunch bimur_bringup bimur_right_ur5_arm_gazebo_1_moveit.launch` <br>
UR5 Arm + Gripper: <br>
`roslaunch bimur_bringup bimur_right_robot_gazebo_1_moveit.launch`

### Manipulation:
```
roslaunch bimur_bringup bimur_right_robot_gazebo_1_moveit.launch
rosrun bimur_manipulation gazebo_moveit_example_1.py
rosrun bimur_manipulation gazebo_moveit_example_2.py
```

## Real Robot

In the teach pendant on the robot, select Program Robot > Load Program > Open ur_driver.upr > Press play <br>

`roslaunch bimur_bringup bimur_right_robot_real_moveit.launch robot_ip:=172.22.22.2 kinematics_config:="$(rospack find bimur_ur_launch)/etc/bimur_right_arm_calibration.yaml" gripper_test:=true`

### Manipulation:
```
roslaunch bimur_bringup bimur_right_robot_real_moveit.launch robot_ip:=172.22.22.2
rosrun bimur_manipulation real_actionclient_example.py
rosrun bimur_manipulation real_moveit_example.py
```

### Start astra camera
- Make sure the camera is plugged into the computer vis USB <br>
- Install camera driver: https://github.com/orbbec/ros_astra_camera <br>
`roslaunch astra_launch astra.launch`

### Start microphone
- Make sure the ReSpeaker Microphone Array is plugged into the computer vis USB <br>
- Check audio device name: `arecord -l` <br>
- Install microphone driver: https://github.com/furushchev/respeaker_ros <br>
`roslaunch respeaker_ros respeaker.launch`

## To perform 7 behaviors (Look, grasp, pick, hold, shake, lower, drop)
`rosrun bimur_manipulation real_behaviors.py`

## To perform push behavior (Arguments for -pt: slow, fast)
`rosrun bimur_manipulation real_behaviors.py -pt slow`
