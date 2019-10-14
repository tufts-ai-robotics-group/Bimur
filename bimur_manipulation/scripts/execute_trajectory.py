#!/usr/bin/env python
import rospy
import moveit_commander
import tf
import rospkg
import os

from gazebo_msgs.srv import (
    SpawnModel,
    DeleteModel,
)

from geometry_msgs.msg import (
    PoseStamped,
    Pose,
    Point,
    Quaternion,
)

rospy.init_node('execute_trajectory', anonymous=True)

robot = moveit_commander.RobotCommander()
arm_group = moveit_commander.MoveGroupCommander("manipulator")
grp_group = moveit_commander.MoveGroupCommander("gripper")

# arm_group.set_planner_id("LBKPIECE")

# You can get the reference frame for a certain group by executing this line:
print "Arm Reference frame: %s" % arm_group.get_planning_frame()
print "Gripper Reference frame: %s" % grp_group.get_planning_frame()

# You can get the end-effector link for a certaing group executing this line:
print "Arm End effector: %s" % arm_group.get_end_effector_link()
print "Gripper End effector: %s" % grp_group.get_end_effector_link()

# You can get a list with all the groups of the robot like this:
print "Robot Groups:"
print robot.get_group_names()

# You can get the current values of the joints like this:
print "Arm Current Joint Values:"
print arm_group.get_current_joint_values()
print "Gripper Current Joint Values:"
print grp_group.get_current_joint_values()

# You can also get the current Pose of the end-effector of the robot like this:
print "Arm Current Pose:"
print arm_group.get_current_pose()

# Finally, you can check the general status of the robot like this:
print "Robot State:"
print robot.get_current_state()


def spawn_gazebo_model(model_path, model_name, model_pose, reference_frame="world"):
  """
  Spawn model in gazebo
  """
  model_xml = ''
  with open(model_path, "r") as model_file:
    model_xml = model_file.read().replace('\n', '')
  rospy.wait_for_service('/gazebo/spawn_urdf_model')
  try:
    spawn_urdf = rospy.ServiceProxy('/gazebo/spawn_urdf_model', SpawnModel)
    resp_urdf = spawn_urdf(model_name, model_xml, "/", model_pose, reference_frame)
  except rospy.ServiceException, e:
    rospy.logerr("Spawn URDF service call failed: {0}".format(e))

def delete_gazebo_model(models):
  """
  Delete model in gazebo
  """
  try:
    delete_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
    for a_model in models:
      resp_delete = delete_model(a_model)
  except rospy.ServiceException, e:
    rospy.loginfo("Delete Model service call failed: {0}".format(e))

rospack = rospkg.RosPack()
pack_path = rospack.get_path('bimur_description')


block_path = pack_path+os.sep+'objects'+os.sep+'block.urdf'
block_name = 'block'
block_pose = Pose(position=Point(x=0.4, y=0.05, z=0.75))

delete_gazebo_model([block_name])
spawn_gazebo_model(block_path, block_name, block_pose)

arm_group.set_named_target('zero')
arm_group.go(wait=True)
print("Point 1")

pose = arm_group.get_current_pose().pose

# Block point
pose.position.x = 0.4
pose.position.y = 0.0
pose.position.z = 0.90

downOrientation = tf.transformations.quaternion_from_euler(0, 3.1415/2, 0)
print("downOrientation: ", downOrientation)
pose.orientation.x = downOrientation[0]
pose.orientation.y = downOrientation[1]
pose.orientation.z = downOrientation[2]
pose.orientation.w = downOrientation[3]

arm_group.set_pose_target(pose)
arm_group.go(wait=True)
print("Point 2")

# Close
grp_group.set_joint_value_target([0.8039005131791948, -0.8039005131791948, 0.8039005131791948, 0.8039005131791948, -0.8039005131791948, 0.8039005131791948])
#grp_group.set_named_target('close')
grp_group.go(wait=True)
print("Point 3")

pose = arm_group.get_current_pose().pose
pose.position.x += 0.1
pose.position.y += 0
pose.position.z += 0
arm_group.set_pose_target(pose)
arm_group.go(wait=True)
print("Point 4")

# Open
#grp_group.set_joint_value_target([9.800441184282249e-05, -9.800441184282249e-05, 9.800441184282249e-05, 9.800441184282249e-05, -9.800441184282249e-05, 9.800441184282249e-05])
grp_group.set_named_target('open')
grp_group.go(wait=True)
print("Point 5")

arm_group.set_named_target('home')
arm_group.go(wait=True)
print("Point 6")

arm_group.set_joint_value_target([-0.21957805043352518, -1.097296859939564, 1.8945345194815335,
                            -2.366067038969164, -1.571228181260084, -1.0061550793898952])
arm_group.go(wait=True)
print("Point 7")
