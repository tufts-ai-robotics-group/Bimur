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
PointA = geometry_msgs.msg.Pose(orientation=geometry_msgs.msg.Quaternion(x = 0.475172464399, y = 0.514251804826, z =  -0.498473726855, w = 0.51115570421), 
		                                position=geometry_msgs.msg.Point(x = 0.194169087101, y = 0.0756737873705, z = 1.04618175164))

# name: [shoulder_pan_joint, shoulder_lift_joint, elbow_joint, wrist_1_joint, wrist_2_joint,
#   wrist_3_joint, gripper_finger1_joint, gripper_finger2_joint, gripper_finger1_inner_knuckle_joint,
#   gripper_finger2_inner_knuckle_joint, gripper_finger1_finger_tip_joint, gripper_finger2_finger_tip_joint]
# position: [-0.2944224516498011, -1.0997055212603968, 1.9248828887939453, -2.1016886870013636, -4.0012221972094935, 0.3613124191761017, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# velocity: [0.0, 0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0, -0.0]
# effort: [1.9929696321487427, 1.8831208944320679, 1.6387635469436646, 0.10827792435884476, 0.08235222101211548, -0.1296284943819046, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

PointB = geometry_msgs.msg.Pose(orientation=geometry_msgs.msg.Quaternion(x = 0.511069134518, y = 0.493728479455, z = -0.493499229966, w = 0.501496797937), 
		                                position=geometry_msgs.msg.Point(x = 0.455492635233, y = 0.0733242723776, z = 0.978839379589))

# name: [shoulder_pan_joint, shoulder_lift_joint, elbow_joint, wrist_1_joint, wrist_2_joint,
#   wrist_3_joint, gripper_finger1_joint, gripper_finger2_joint, gripper_finger1_inner_knuckle_joint,
#   gripper_finger2_inner_knuckle_joint, gripper_finger1_finger_tip_joint, gripper_finger2_finger_tip_joint]
# position: [-0.642334286366598, -0.7638228575335901, 1.697338581085205, -1.950127903615133, -4.086104933415548, 0.8312606811523438, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# velocity: [0.0, 0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0, -0.0]
# effort: [2.7013816833496094, 2.2104251384735107, 1.3428444862365723, 0.11590313166379929, -0.10217776149511337, -0.0930275097489357, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


PointC = geometry_msgs.msg.Pose(orientation=geometry_msgs.msg.Quaternion(x = 0.479738315689, y = 0.497000080917, z = -0.495447868523, w = 0.526662584207), 
		                                position=geometry_msgs.msg.Point(x = 0.728683963508, y = 0.0751983345402, z = 1.05618339651))

# name: [shoulder_pan_joint, shoulder_lift_joint, elbow_joint, wrist_1_joint, wrist_2_joint,
#   wrist_3_joint, gripper_finger1_joint, gripper_finger2_joint, gripper_finger1_inner_knuckle_joint,
#  gripper_finger2_inner_knuckle_joint, gripper_finger1_finger_tip_joint, gripper_finger2_finger_tip_joint]
# position: [-0.8076985518084925, -0.18340903917421514, 1.2241120338439941, -1.9431269804583948, -4.231040302907125, 0.9479743242263794, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# velocity: [0.0, 0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0, -0.0]
# effort: [2.820197582244873, 3.4994661808013916, 1.416824221611023, 0.2363813817501068, -0.03965106979012489, -0.14487890899181366, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

PointW = geometry_msgs.msg.Pose(orientation=geometry_msgs.msg.Quaternion(x = 0.504368421729, y = 0.511398877174, z = -0.500171283189, w = 0.483644881148), 
		                                position=geometry_msgs.msg.Point(x = 0.486415571154, y = 0.0844080537887, z = 1.23171086284))

# name: [shoulder_pan_joint, shoulder_lift_joint, elbow_joint, wrist_1_joint, wrist_2_joint,
#   wrist_3_joint, gripper_finger1_joint, gripper_finger2_joint, gripper_finger1_inner_knuckle_joint,
#   gripper_finger2_inner_knuckle_joint, gripper_finger1_finger_tip_joint, gripper_finger2_finger_tip_joint]
# position: [-0.4246581236468714, -0.34343129793276006, 1.7390179634094238, -2.627223793660299, -4.015395704899923, 0.5452092289924622, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# velocity: [0.0, 0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0, -0.0]
# effort: [2.5982584953308105, 2.470475196838379, 1.0648599863052368, 0.035075947642326355, 0.15555420517921448, 0.10522784292697906, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

#What to do with the Gripper
GripperClose = robotiq_85_msgs.msg.GripperCmd(position=0.0, force = 31.0)

GripperOpen = robotiq_85_msgs.msg.GripperCmd(position= 255.0, force = 31.0)

#How long to wait before and after commanding the gripper
WaitA = (2, 2)

WaitB = (2, 2)

WaitC = (2, 2)


Waypoints = [(PointA, GripperClose, WaitA), (PointW, GripperClose, WaitA), (PointB, GripperClose, WaitB), (PointW, GripperClose, WaitA), (PointC, GripperClose, WaitC), (PointW, GripperClose, WaitA)]

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