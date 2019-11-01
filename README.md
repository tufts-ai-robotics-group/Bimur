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

## Real Robot:

`roslaunch ur_modern_driver ur5_bringup.launch robot_ip:=172.22.22.2`

# Robotiq Gripper

`roslaunch bimur_bringup robotiq_gripper.launch`

https://github.com/ros-industrial/robotiq
http://wiki.ros.org/robotiq/Tutorials/Control%20of%20a%202-Finger%20Gripper%20using%20the%20Modbus%20RTU%20protocol%20%28ros%20kinetic%20and%20newer%20releases%29
