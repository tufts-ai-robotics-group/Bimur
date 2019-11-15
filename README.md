# Bimur
Bimur (Bi-manual UR5). This repository is for packages related to UR5 at AIR Lab, Tufts.

<img src="pics/Bimur.png" align="middle">

# Set up Repository:
`git clone --recurse-submodules -j8 https://github.com/tufts-ai-robotics-group/Bimur/tree/faizan-exp` 

# Right Arm

## Gazebo + MoveIt Rviz:

`roslaunch bimur_bringup bimur_right_arm_gazebo_moveit.launch`

## Manipulation:

```
roslaunch bimur_bringup bimur_right_arm_gazebo.launch
rosrun bimur_manipulation execute_trajectory.py
```

## Real Robot:

`roslaunch bimur_bringup right_arm.launch`
