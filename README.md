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

## TODO:
- The arm can be controlled by Moveit but the gripper needs more work
- The controller for gripper is set up but it expects an action server 'gripper/gripper_command' that can execute commands based on the 'control_msgs' format
- The gripper at the moment only interacts through the topics provided by the driver so the action server just needs to tunnel requests to the topic
- Implementing that and then potentially debugging any related issues would provide pick and place through MoveIt
