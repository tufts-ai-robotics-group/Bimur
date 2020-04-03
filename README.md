# Bimur
Bimur (Bi-manual UR5). This repository is for packages related to UR5 at AIR Lab, Tufts.

<img src="pics/Bimur.png" align="middle">

# Install:
`git clone https://github.com/tufts-ai-robotics-group/Bimur.git`

## Submodules

Following GitHub repo. was used (no need to clone them):
```
git clone https://github.com/UniversalRobots/Universal_Robots_ROS_Driver.git
git checkout 2ee8f2b

git clone -b calibration_devel https://github.com/fmauch/universal_robot.git
git checkout e5a176c

git clone https://github.com/StanleyInnovation/robotiq_85_gripper
git checkout 2240a8c

git clone https://github.com/ros-industrial/robotiq.git
git checkout 66961ec
```

# Right Robot

## Rviz
Launch URDF in Rviz for visualization <br>
`roslaunch bimur_description bimur_right_robot_rviz.launch`

## Gazebo
Launch URDF in Gazebo with specific joint poses: <br>
`roslaunch bimur_bringup bimur_right_robot_gazebo_1.launch`

Launch URDF in Gazebo with specific kinematics_config: <br>
`roslaunch bimur_bringup bimur_right_robot_gazebo_2.launch`

## Gazebo + MoveIt Rviz:
UR5 Arm: <br>
`roslaunch bimur_bringup bimur_right_ur5_arm_gazebo_1_moveit.launch`
UR5 Arm + Gripper: <br>
`roslaunch bimur_bringup bimur_right_robot_gazebo_1_moveit.launch`

## Real Robot
`roslaunch bimur_bringup bimur_right_robot_real_moveit.launch robot_ip:=172.22.22.2 kinematics_config:="$(rospack find bimur_ur_launch)/etc/bimur_right_arm_calibration.yaml" gripper_test:=true`

### Manipulation:
```
roslaunch bimur_bringup bimur_right_robot_real_moveit.launch robot_ip:=172.22.22.2 kinematics_config:="$(rospack find bimur_ur_launch)/etc/bimur_right_arm_calibration.yaml" gripper_test:=false
rosrun bimur_manipulation trajectory_test.py
```
==============================
OLD:
```
roslaunch bimur_bringup bimur_right_arm_gazebo.launch
rosrun bimur_manipulation execute_trajectory.py
```
