#!/usr/bin/env python2

import sys
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
import robotiq_85_msgs.msg

ARM_GROUP_NAME = "manipulator"
PLANNER_ID = "TRRT"
PLANNING_TIME = 10.0

GRIPPER_TOPIC = "gripper/cmd"

GRIPPER_WAIT_BEFORE = 2
GRIPPER_WAIT_AFTER = 2
WAYPOINT_WAIT = 1

Robot = None
Scene = None
Group = None
GripperPub = None

Planning_Frame = None
End_Effector_Link = None

PointL = geometry_msgs.msg.Pose(orientation=geometry_msgs.msg.Quaternion(x = 0.500840693262, y = 0.494930032374, z =  -0.496034894632, w = 0.508086849204), 
		                                position=geometry_msgs.msg.Point(x = 0.186971272096, y = 0.0705390708275, z = 1.06040743537))


PointM = geometry_msgs.msg.Pose(orientation=geometry_msgs.msg.Quaternion(x = 0.504327545037, y = 0.502590730078, z = -0.481190402774, w = 0.511382519876), 
		                                position=geometry_msgs.msg.Point(x = 0.454134457677, y = 0.0698302440452, z = 0.983767220427))

PointR = geometry_msgs.msg.Pose(orientation=geometry_msgs.msg.Quaternion(x = 0.521736343569, y = 0.503875825395, z = -0.476069634962, w = 0.497250483207), 
		                                position=geometry_msgs.msg.Point(x = 0.725216225868, y = 0.0645384454211, z = 1.06671386315))

PointLPrime = geometry_msgs.msg.Pose(orientation=geometry_msgs.msg.Quaternion(x = 0.500954530225, y = 0.462608328887, z = -0.527722776423, w = 0.506504455987), 
		                                position=geometry_msgs.msg.Point(x = 0.183537981128, y = 0.0951741145907, z = 1.13729980784))

PointMPrime = geometry_msgs.msg.Pose(orientation=geometry_msgs.msg.Quaternion(x = 0.496286872676, y = 0.500783319307, z = -0.5018127301, w = 0.501098185013), 
		                                position=geometry_msgs.msg.Point(x = 0.456530077334, y = 0.0811008011589, z = 1.14672215135))

PointRPrime = geometry_msgs.msg.Pose(orientation=geometry_msgs.msg.Quaternion(x = 0.543185405826, y = 0.491713792398, z = -0.487982129957, w = 0.474384445472), 
		                                position=geometry_msgs.msg.Point(x = 0.718521375649, y = 0.0468752959776, z = 1.15662933133))


# Waypoints to get to different places from MPrime
ToPointL = [PointLPrime, PointL]
ToPointM = [PointM]
ToPointR = [PointRPrime, PointR]

# Waypoints to get from different places to MPrime
FromPointL = [PointLPrime, PointMPrime]
FromPointM = [PointMPrime]
FromPointR = [PointRPrime, PointMPrime]

# Pick to Place Destinations
PickPlaceDecisions = {"left": "right", "middle" : "right", "right" : "left"}
Locations = {"left": (ToPointL, FromPointL), "middle": (ToPointM, FromPointM), "right": (ToPointR, FromPointR)}

def getBallLocation():
	# "left"
	# "right"
	# "middle"

	return "right"

def moveBall():
	pos = getBallLocation()
	targetPos = PickPlaceDecisions[pos]

	setGripper(31.0, 255.0)

	# Pick
	moveWaypoints(Locations[pos][0])
	setGripper(31.0, 0.0)
	moveWaypoints(Locations[pos][1])

	# Place
	moveWaypoints(Locations[targetPos][0])
	setGripper(31.0, 255.0)
	moveWaypoints(Locations[targetPos][1])

	# Check
	finalPos = getBallLocation()

	return (finalPos == targetPos)

def moveWaypoints(wp):
	for w in wp:
		Group.set_pose_target(w)
		plan = Group.plan()
		Group.go()
		rospy.sleep(WAYPOINT_WAIT)

def setGripper(f, p):
	msg = robotiq_85_msgs.msg.GripperCmd(position=p, force = f)
	rospy.sleep(GRIPPER_WAIT_BEFORE)
	GripperPub.publish(msg)
	rospy.sleep(GRIPPER_WAIT_AFTER)

if __name__ == "__main__":
	moveit_commander.roscpp_initialize(sys.argv)
	rospy.init_node('move_group_python_interface',
                anonymous=True)

	Robot = moveit_commander.RobotCommander()
	Scene = moveit_commander.PlanningSceneInterface()
	Group = moveit_commander.MoveGroupCommander(ARM_GROUP_NAME)

	Group.set_planner_id(PLANNER_ID)
	Group.set_planning_time(PLANNING_TIME)

	Planning_Frame = Group.get_planning_frame()
	End_Effector_Link = Group.get_end_effector_link()

	GripperPub = rospy.Publisher(GRIPPER_TOPIC, robotiq_85_msgs.msg.GripperCmd, queue_size=1)

	# print("Planning_Frame: ", Planning_Frame)
	# print("End_Effector_Link:", End_Effector_Link)

	while not rospy.is_shutdown():
		v = moveBall()