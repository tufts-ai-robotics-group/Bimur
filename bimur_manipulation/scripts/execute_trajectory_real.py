#!/usr/bin/env python2

import sys
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
import robotiq_85_msgs.msg

ARM_GROUP_NAME = "manipulator"

Robot = None
Scene = None
Group = None

Planning_Frame = None
End_Effecor_Link = None

#Where to position the Gripper
PointA = geometry_msgs.msg.Pose(orientation=geometry_msgs.msg.Quaternion(x = -0.0606165997727, y = 0.708775679025, z =  -0.0103372290094, w = 0.702748750515), 
		                                 position=geometry_msgs.msg.Point(x = 0.188000468541, y = 0.247979602258, z = 1.05076819189))

PointB = geometry_msgs.msg.Pose(orientation=geometry_msgs.msg.Quaternion(x = -0.063018706213, y = 0.692740996044, z = -0.0052749862298, w = 0.718408469874), 
		                                 position=geometry_msgs.msg.Point(x = 0.452787957968, y = 0.251511110693, z = 0.984304920617))

PointC = geometry_msgs.msg.Pose(orientation=geometry_msgs.msg.Quaternion(x = 0.0205839648497, y = 0.695121505723, z = 0.0184112803517, w = 0.718361620236), 
		                                 position=geometry_msgs.msg.Point(x = 0.725459939611, y = 0.220261723827, z = 1.06756277814))

#What to do with the Gripper
GripA = robotiq_85_msgs.msg.GripperCmd()

GripB = robotiq_85_msgs.msg.GripperCmd()

GripC = robotiq_85_msgs.msg.GripperCmd()

#How long to wait before and after commanding the gripper
WaitA = (1, 1)

WaitB = (1, 1)

WaitC = (1, 1)


Waypoints = [(PointA, GripA, WaitA), (PointB, GripB, WaitB), (PointC, GripC, WaitC)]

def cycle_waypoints():
	index = 0

	while(not rospy.is_shutdown()):
		target = Waypoints[index] 

		Group.set_pose_target(target[0])
		plan = Group.plan()

		Group.go()

		rospy.sleep(target[2][0])

		#TODO: Grip

		rospy.sleep(target[2][1])

		index = (index + 1) % len(Waypoints)

if __name__ == "__main__":
	moveit_commander.roscpp_initialize(sys.argv)
	rospy.init_node('move_group_python_interface',
                anonymous=True)

	Robot = moveit_commander.RobotCommander()
	Scene = moveit_commander.PlanningSceneInterface()
	Group = moveit_commander.MoveGroupCommander(ARM_GROUP_NAME)


	Planning_Frame = Group.get_planning_frame()
	End_Effector_Link = Group.get_end_effector_link()

	# print("Planning_Frame: ", Planning_Frame)
	# print("End_Effector_Link:", End_Effector_Link)

	cycle_waypoints()