#!/usr/bin/env python

# Gyan Tatiya

import os
import subprocess
import time
from datetime import datetime

import rospy
import moveit_commander
import tf
import rospkg

from robotiq_85_msgs.msg import GripperCmd

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


def start_rosbag_recording(path, filename, topic_list_, num=None):

    rospy.loginfo(rospy.get_name() + ' start rosbag recording: ' + str(topic_list_))

    if not os.path.exists(path):
        os.mkdir(path)

    rosbag_record_cmd = 'rosbag record -O ' + filename + ' '
    for topic_name in topic_list_:
        rosbag_record_cmd += topic_name + ' '
    if num:
        rosbag_record_cmd += '-l ' + str(num)

    rosbag_process = subprocess.Popen(rosbag_record_cmd, stdin=subprocess.PIPE, shell=True, cwd=path)

    return rosbag_process


def stop_rosbag_recording():

    s = "/record"
    list_cmd = subprocess.Popen("rosnode list", shell=True, stdout=subprocess.PIPE)
    list_output = list_cmd.stdout.read()
    retcode = list_cmd.wait()
    assert retcode == 0, "List command returned %d" % retcode
    for str in list_output.split("\n"):
        if (str.startswith(s)):
            os.system("rosnode kill " + str)

    rospy.loginfo(rospy.get_name() + ' stop rosbag recording')


class Behaviors(object):
    def __init__(self, sensor_data_path_):

        print("Initializing node... ")
        rospy.init_node('real_behaviors', anonymous=True)
        print("Running. Ctrl-c to quit")

        self.sensor_data_path = sensor_data_path_
        if not os.path.exists(self.sensor_data_path):
            os.makedirs(self.sensor_data_path)

        # Sensory data to be saved:
        haptic_topic = "/joint_states"
        force_topic = "/wrench"
        gripper_topic = "/gripper/joint_states"
        image_head_rgb_topic = "/camera/rgb/image_raw"
        image_head_depth_topic = "/camera/depth/image_raw"
        audio_topic = "/audio"

        self.topic_list = [haptic_topic, force_topic, gripper_topic, image_head_rgb_topic, image_head_depth_topic,
                           audio_topic]

        # Sensory data to be saved in a bag file:
        self.point_cloud_topic = '/camera/depth_registered/points'

        robot = moveit_commander.RobotCommander()
        self.arm_group = moveit_commander.MoveGroupCommander("manipulator")  # Creating moveit client to control arm
        self.gripper_pub = rospy.Publisher('/gripper/cmd', GripperCmd, queue_size=1)

        self.initialise_robot()

    def open_gripper(self):

        msg = GripperCmd(position=0.1, speed=1, force=100.0)
        rospy.sleep(1)
        self.gripper_pub.publish(msg)
        rospy.sleep(1)

    def close_gripper(self):

        msg = GripperCmd(position=0.00, speed=1, force=100.0)
        rospy.sleep(1)
        self.gripper_pub.publish(msg)
        rospy.sleep(1)

    def initialise_robot(self):

        print("\nInitialising robot pose")

        self.open_gripper()
        self.arm_group.set_named_target('ready')
        self.arm_group.go(wait=True)

    def look(self):

        print("\nLooking...")
        behavior = "ur5_1-look"
        num_msg_to_record = 5

        self.initialise_robot()

        bag_filename = '{0}_{1}_sensor_data'.format(behavior, datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        topic_list = ['/camera/depth_registered/points', '/camera/depth/image_raw', '/camera/rgb/image_raw']
        start_rosbag_recording(self.sensor_data_path, bag_filename, topic_list, num=num_msg_to_record)

        time.sleep(1.0)

    def grasp(self):

        print("\nGrasping...")
        behavior = "ur5_2-grasp"
        self.open_gripper()

        self.initialise_robot()

        bag_filename = '{0}_{1}_pointcloud'.format(behavior, datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        p = start_rosbag_recording(self.sensor_data_path, bag_filename, topic_list_=[self.point_cloud_topic], num=5)

        bag_filename = '{0}_{1}_sensor_data'.format(behavior, datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        start_rosbag_recording(self.sensor_data_path, bag_filename, self.topic_list, num=None)
        time.sleep(2.0)

        point = [-0.90, -0.98, 2.10, -2.01, 2.02, 2.59]
        self.arm_group.set_joint_value_target(point)
        self.arm_group.go(wait=True)

        self.close_gripper()

        stop_rosbag_recording()

    def pick(self):

        print("\nPicking...")
        behavior = "ur5_3-pick"

        bag_filename = '{0}_{1}_pointcloud'.format(behavior, datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        p = start_rosbag_recording(self.sensor_data_path, bag_filename, topic_list_=[self.point_cloud_topic], num=5)

        bag_filename = '{0}_{1}_sensor_data'.format(behavior, datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        start_rosbag_recording(self.sensor_data_path, bag_filename, self.topic_list, num=None)
        time.sleep(2.0)

        self.close_gripper()
        point = [-0.76, -0.70, 2.29, -2.57, 2.14, 2.50]
        self.arm_group.set_joint_value_target(point)
        self.arm_group.go(wait=True)

        stop_rosbag_recording()

    def hold(self):

        print("\nHolding...")
        behavior = "ur5_4-hold"

        point = [-0.63, -1.49, 2.23, -2.76, 4.09, 1.86]
        self.arm_group.set_joint_value_target(point)
        self.arm_group.go(wait=True)

        bag_filename = '{0}_{1}_pointcloud'.format(behavior, datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        p = start_rosbag_recording(self.sensor_data_path, bag_filename, topic_list_=[self.point_cloud_topic], num=5)

        bag_filename = '{0}_{1}_sensor_data'.format(behavior, datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        start_rosbag_recording(self.sensor_data_path, bag_filename, self.topic_list, num=None)
        time.sleep(2.0)

        stop_rosbag_recording()

    def shake(self):

        print("\nShaking...")
        behavior = "ur5_5-shake"

        bag_filename = '{0}_{1}_pointcloud'.format(behavior, datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        p = start_rosbag_recording(self.sensor_data_path, bag_filename, topic_list_=[self.point_cloud_topic], num=5)

        bag_filename = '{0}_{1}_sensor_data'.format(behavior, datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        start_rosbag_recording(self.sensor_data_path, bag_filename, self.topic_list, num=None)

        for _ in range(3):
            # point = [-1.16, -1.85, 2.44, -2.86, 4.45, -1.65]  # Too far from camera
            point = [-0.79, -1.61, 2.77, -3.24, 4.18, -1.50]  # Close to camera
            self.arm_group.set_joint_value_target(point)
            self.arm_group.go(wait=True)

            point = [-0.50, -1.38, 2.41, -2.98, 4.02, 2.05]  # Far to camera
            self.arm_group.set_joint_value_target(point)
            self.arm_group.go(wait=True)

        stop_rosbag_recording()

    def lower(self):

        print("\nLowering...")
        behavior = "ur5_6-lower"

        # point = [-0.48, -0.25, 2.23, -3.08, -4.00, 2.24]
        point = [-0.63, -0.49, 2.29, -2.82, 2.18, 3.91]
        # point = [-0.48, -0.30, 2.24, -3.08, 2.28, 2.23]
        self.arm_group.set_joint_value_target(point)
        self.arm_group.go(wait=True)

        bag_filename = '{0}_{1}_pointcloud'.format(behavior, datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        p = start_rosbag_recording(self.sensor_data_path, bag_filename, topic_list_=[self.point_cloud_topic], num=5)

        bag_filename = '{0}_{1}_sensor_data'.format(behavior, datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        start_rosbag_recording(self.sensor_data_path, bag_filename, self.topic_list, num=None)
        time.sleep(2.0)

        # point = [-0.83, -0.84, 2.30, -2.34, -4.19, 2.62]
        # point = [-0.86, -0.92, 2.21, -2.18, 2.06, 4.15]
        # point = [-0.82, -0.88, 2.19, -2.19, 2.05, 2.5]
        point = [-0.84, -0.88, 2.18, -2.19, 2.06, 2.58]
        self.arm_group.set_joint_value_target(point)
        self.arm_group.go(wait=True)

        self.close_gripper()

        stop_rosbag_recording()

    def drop(self):

        print("\nDropping...")
        behavior = "ur5_7-drop"

        bag_filename = '{0}_{1}_pointcloud'.format(behavior, datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        p = start_rosbag_recording(self.sensor_data_path, bag_filename, topic_list_=[self.point_cloud_topic], num=5)

        bag_filename = '{0}_{1}_sensor_data'.format(behavior, datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        start_rosbag_recording(self.sensor_data_path, bag_filename, self.topic_list, num=None)

        time.sleep(3.0)
        self.open_gripper()
        time.sleep(3.0)

        stop_rosbag_recording()

    def push(self):

        print("\nPushing...")
        behavior = "ur5_8-push"

        msg = GripperCmd(position=0.02, speed=1, force=100.0)
        rospy.sleep(1)
        self.gripper_pub.publish(msg)
        rospy.sleep(1)

        point = [-0.84, -1.19, 2.19, -1.91, 2.07, 2.59]
        self.arm_group.set_joint_value_target(point)
        self.arm_group.go(wait=True)

        bag_filename = '{0}_{1}_pointcloud'.format(behavior, datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        p = start_rosbag_recording(self.sensor_data_path, bag_filename, topic_list_=[self.point_cloud_topic], num=5)

        bag_filename = '{0}_{1}_sensor_data'.format(behavior, datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        start_rosbag_recording(self.sensor_data_path, bag_filename, self.topic_list, num=None)
        time.sleep(2.0)

        point = [-0.99, -0.83, 2.01, -2.04, 1.94, 2.77]
        self.arm_group.set_joint_value_target(point)
        self.arm_group.go(wait=True)
        time.sleep(2.0)

        stop_rosbag_recording()


if __name__ == "__main__":

    """
    This script performs look, grasp, pick, hold, shake, lower, drop, push behaviors.
    And saves sensory data:
        /joint_states, /wrench, /gripper/joint_states, /camera/rgb/image_raw, /camera/depth/image_raw, /audio
        /camera/depth_registered/points
    """

    sensor_data_path = r"/media/gyan/My Passport/UR5_Dataset_Temp/"

    object_name = 'obj1'
    trial_no = 0
    sensor_data_path += os.sep + "ur5_" + object_name + os.sep + "trial-" + str(trial_no) + "_" + \
                        datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    b = Behaviors(sensor_data_path)
    b.initialise_robot()

    b.look()
    b.grasp()
    b.pick()
    b.hold()
    b.shake()
    b.lower()
    b.drop()
    b.push()

    """
    TODO:
    
    Fast push
    """
