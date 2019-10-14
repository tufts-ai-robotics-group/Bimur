# Bimur
Bimur (Bi-manual UR5). This repository is for packages related to UR5 at AIR Lab, Tufts.

<img src="pics/Bimur.png" align="middle">

# Right Arm

## Gazebo + MoveIt Rviz:

`roslaunch bimur_bringup bimur_right_arm_gazebo_moveit.launch`

## Manipulation:

```
roslaunch bimur_bringup bimur_right_arm_gazebo.launch
rosrun bimur_manipulation execute_trajectory.py
```
