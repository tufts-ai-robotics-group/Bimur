# Bimur
Bimur (Bi-manual UR5). This repository is for packages related to UR5 at AIR Lab, Tufts.

<img src="pics/Bimur.png" align="middle">

# Installation

`git clone https://github.com/tufts-ai-robotics-group/Bimur.git`

## Requirements

1. Ubuntu 16.04
2. ROS Kinetic
3. Gazebo 7.x

## ROS Packages

```
sudo apt-get install ros-kinetic-control-toolbox
sudo apt-get install ros-kinetic-controller-interface
sudo apt-get install ros-kinetic-controller-manager-msgs
sudo apt-get install ros-kinetic-controller-manager
sudo apt-get install ros-kinetic-effort-controllers
sudo apt-get install ros-kinetic-force-torque-sensor-controller
sudo apt-get install ros-kinetic-gazebo-ros-control
sudo apt-get install ros-kinetic-industrial-robot-status-interface
sudo apt-get install ros-kinetic-joint-state-controller
sudo apt-get install ros-kinetic-joint-state-publisher-gui
sudo apt-get install ros-kinetic-joint-trajectory-controller
sudo apt-get install ros-kinetic-moveit
sudo apt-get install ros-kinetic-moveit-core
sudo apt-get install ros-kinetic-moveit-kinematics
sudo apt-get install ros-kinetic-moveit-ros-visualization
sudo apt-get install ros-kinetic-position-controllers
sudo apt-get install ros-kinetic-rqt-joint-trajectory-controller
sudo apt-get install ros-kinetic-socketcan-interface
sudo apt-get install ros-kinetic-soem
sudo apt-get install ros-kinetic-ur-client-library
sudo apt-get install ros-kinetic-velocity-controllers

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
