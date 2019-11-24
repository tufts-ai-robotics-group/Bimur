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

Robot = None
Scene = None
Group = None
GripperPub = None

Planning_Frame = None
End_Effector_Link = None

#Where to position the Gripper
PointA = geometry_msgs.msg.Pose(orientation=geometry_msgs.msg.Quaternion(x = -0.00634722655519, y = 0.726333097936, z =  0.000507039619319, w = 0.687313383012), 
		                                position=geometry_msgs.msg.Point(x = 0.19276567638, y = 0.0658832703881, z = 1.08314827424))

#name: [shoulder_pan_joint, shoulder_lift_joint, elbow_joint, wrist_1_joint, wrist_2_joint, wrist_3_joint]
#position: [-3.0008116404162806, -2.178382698689596, -1.9325106779681605, -0.4876349608050745, -0.8477738539325159, -1.7608187834369105]
#velocity: [0.0, 0.0, -0.0, 0.0, 0.0, 0.0]
#effort: [2.457024335861206, -2.571356773376465, -0.6299487352371216, -0.33703410625457764, -0.0732019767165184, -0.09912768006324768]

PointB = geometry_msgs.msg.Pose(orientation=geometry_msgs.msg.Quaternion(x = -0.0110589476721, y = 0.717402156623, z = 0.0435478508542, w = 0.695208911073), 
		                                position=geometry_msgs.msg.Point(x = 0.457577324366, y = 0.0676122058283, z = 1.01283873588))

#name: [shoulder_pan_joint, shoulder_lift_joint, elbow_joint, wrist_1_joint, wrist_2_joint, wrist_3_joint]
#position: [-3.4023752848254603, -2.4839757124530237, -1.7207120100604456, -0.6877720991717737, -0.8351882139789026, -1.3360899130450647]
#velocity: [0.0, 0.0, -0.0, 0.0, 0.0, 0.0]
#effort: [2.557905912399292, -2.383044719696045, -0.957253098487854, -0.28060758113861084, -0.06405173242092133, -0.1265784204006195]

PointC = geometry_msgs.msg.Pose(orientation=geometry_msgs.msg.Quaternion(x = 0.0112733729864, y = 0.695310535421, z = -0.014002335886, w = 0.718484589246), 
		                                position=geometry_msgs.msg.Point(x = 0.731598552821, y = 0.0719531664246, z = 1.07845406064))

#name: [shoulder_pan_joint, shoulder_lift_joint, elbow_joint, wrist_1_joint, wrist_2_joint, wrist_3_joint]
#position: [-3.612408224736349, -3.015050713215963, -1.219628636037008, -0.9333089033709925, -0.8652656714068812, -0.8920930067645472]
#velocity: [0.0, 0.0, -0.0, 0.0, 0.0, 0.0]
#effort: [2.2081832885742188, -3.669843912124634, -1.1410815715789795, -0.3522845208644867, -0.007625205907970667, 0.24400658905506134]

#What to do with the Gripper
GripA = robotiq_85_msgs.msg.GripperCmd(position=0.0, force = 31.0)

GripB = robotiq_85_msgs.msg.GripperCmd(position= 255.0, force = 31.0)

GripC = robotiq_85_msgs.msg.GripperCmd(position = 0.0, force = 31.0)

#How long to wait before and after commanding the gripper
WaitA = (2, 2)

WaitB = (2, 2)

WaitC = (2, 2)


Waypoints = [(PointA, GripA, WaitA), (PointB, GripB, WaitB), (PointC, GripC, WaitC)]

def cycle_waypoints():
	index = 0

	while(not rospy.is_shutdown()):
		print("Going to Waypoint ", index)
		target = Waypoints[index] 
		Group.set_pose_target(target[0])

		print("Planning...")
		plan = Group.plan()

		print("Executing...")
		Group.go()

		print("Waiting Pre-Grip...")
		rospy.sleep(target[2][0])

		print("Gripping...")
		GripperPub.publish(target[1])

		print("Waiting Post-Grip...")
		rospy.sleep(target[2][1])

		index = (index + 1) % len(Waypoints)

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

	cycle_waypoints()