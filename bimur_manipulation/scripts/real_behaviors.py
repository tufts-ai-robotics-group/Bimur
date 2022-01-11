#!/usr/bin/env python

# Gyan Tatiya

import argparse
import os
import shutil
import subprocess
import time
from copy import copy
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

import actionlib
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectoryPoint


def start_rosbag_recording(path, filename, topic_list_, num=None):
    rospy.loginfo(rospy.get_name() + ' start rosbag recording: ' + str(topic_list_))

    if not os.path.exists(path):
        os.mkdir(path)

    rosbag_record_cmd = 'rosbag record -O ' + filename + ' '
    for topic_name in topic_list_:
        rosbag_record_cmd += topic_name + ' '
    rosbag_record_cmd += '-b 0 '  # Buffer size 0 = infinite
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


def get_label_from_user(labels):

    labels_id = {}
    if labels == 0:
        print("\nWeights in grams")
        labels = [int(lab[:-1]) for lab in labels]
    for i, label in enumerate(sorted(labels)):
        labels_id[label] = i

    id_ = label = ""
    while id_ not in labels_id.values():
        print("\nLabels: IDs")
        for lab, lab_id in sorted(labels_id.items(), key=lambda x: x[1]):
            print('{0}: {1}'.format(lab, lab_id))

        try:
            id_ = int(raw_input("Enter ID: "))
        except ValueError:
            print("Sorry, I didn't understand that\n")
            continue
        else:
            if id_ not in labels_id.values():
                print("ID you entered is not valid\n")
            else:
                label = list(labels_id.keys())[list(labels_id.values()).index(id_)]
                break

    return label


def find_trial_no(object_name_, path, robot):

    behaviors_trial_count = {}
    trial_count = 0
    for root, subdirs, files in os.walk(path + os.sep + robot + "_" + object_name):
        root_list = root.split(os.sep)
        for filename in files:
            filename, fileext = os.path.splitext(filename)

            if fileext == '.bag' and root_list[-2].split('_')[1] == object_name_ and filename.endswith('sensor_data'):
                behavior = filename.split('_')[1]
                behaviors_trial_count.setdefault(behavior, 0)
                behaviors_trial_count[behavior] += 1
            else:
                continue

    behaviors_trial_count_list = sorted(behaviors_trial_count.items(), key=lambda x: x[1])

    if behaviors_trial_count_list:
        print("\nBehaviors: Trial Count")
        for behavior, trial_count in behaviors_trial_count_list:
            print('{0}: {1}'.format(behavior, trial_count))

        behavior, trial_count = sorted(behaviors_trial_count.items(), key=lambda x: x[1])[0]

    return trial_count


class Trajectory(object):

    def __init__(self):
        self._client = actionlib.SimpleActionClient('/scaled_pos_traj_controller/follow_joint_trajectory',
                                                    FollowJointTrajectoryAction)
        self._goal = FollowJointTrajectoryGoal()
        try:
            self._client.wait_for_server(timeout=rospy.Duration(10.0))
        except rospy.exceptions.ROSException as err:
            rospy.logerr("Timed out waiting for Joint Trajectory"
                         " Action Server to connect. Start the action server"
                         " before running example.")
            rospy.signal_shutdown("Timed out waiting for Action Server")
        self.clear()

    def add_point(self, positions, time_):
        point = JointTrajectoryPoint()
        point.positions = copy(positions)
        point.velocities = [0.0] * len(positions)
        point.accelerations = [0.0] * len(positions)
        point.time_from_start = rospy.Duration(time_)

        self._goal.trajectory.points.append(point)
        self._goal.goal_time_tolerance = rospy.Duration(0.0)

    def start(self):
        self._goal.trajectory.header.stamp = rospy.Time.now()
        self._client.send_goal(self._goal)

    def stop(self):
        self._client.cancel_goal()

    def wait(self, timeout=15.0):
        self._client.wait_for_result(timeout=rospy.Duration(timeout))

    def result(self):
        return self._client.get_result()

    def clear(self):
        self._goal = FollowJointTrajectoryGoal()
        self._goal.goal_time_tolerance = rospy.Time(0.1)
        self._goal.trajectory.joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint",
                                             "wrist_2_joint", "wrist_3_joint"]


class Behaviors(object):
    def __init__(self, sensor_data_path_, robot):

        print("Initializing node... ")
        rospy.init_node('real_behaviors', anonymous=True)
        print("Running. Ctrl-c to quit")

        self.sensor_data_path = sensor_data_path_
        self.robot = robot
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

        self.arm_client = Trajectory()
        rospy.on_shutdown(self.arm_client.stop)

        self.arm_group.set_max_velocity_scaling_factor(1)
        self.arm_group.set_max_acceleration_scaling_factor(1)

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

    def start_rosbag_recording(self, behavior):

        bag_filename = '{0}_{1}_pointcloud'.format(behavior, datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        p = start_rosbag_recording(self.sensor_data_path, bag_filename, topic_list_=[self.point_cloud_topic], num=5)

        bag_filename = '{0}_{1}_sensor_data'.format(behavior, datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        start_rosbag_recording(self.sensor_data_path, bag_filename, self.topic_list, num=None)

    def look(self):

        print("\nLooking...")
        behavior = self.robot + "_1-look"
        num_msg_to_record = 5

        # self.initialise_robot()

        bag_filename = '{0}_{1}_sensor_data'.format(behavior, datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        topic_list = ['/camera/depth_registered/points', '/camera/depth/image_raw', '/camera/rgb/image_raw']
        start_rosbag_recording(self.sensor_data_path, bag_filename, topic_list, num=num_msg_to_record)

        time.sleep(1.0)

    def grasp(self):

        print("\nGrasping...")
        behavior = self.robot + "_2-grasp"
        self.open_gripper()

        # self.initialise_robot()

        self.start_rosbag_recording(behavior)
        time.sleep(2.0)

        point = [-0.90, -0.98, 2.10, -2.01, 2.02, 2.59]
        self.arm_group.set_joint_value_target(point)
        self.arm_group.go(wait=True)

        self.close_gripper()

        stop_rosbag_recording()

    def pick(self):

        print("\nPicking...")
        behavior = self.robot + "_3-pick"

        self.start_rosbag_recording(behavior)
        time.sleep(2.0)

        self.close_gripper()
        point = [-0.76, -0.70, 2.29, -2.57, 2.14, 2.50]
        self.arm_group.set_joint_value_target(point)
        self.arm_group.go(wait=True)

        stop_rosbag_recording()

    def hold(self):

        print("\nHolding...")
        behavior = self.robot + "_4-hold"

        point = [-0.63, -1.49, 2.23, -2.76, 4.09, 1.86]
        self.arm_group.set_joint_value_target(point)
        self.arm_group.go(wait=True)

        self.start_rosbag_recording(behavior)
        time.sleep(2.0)

        stop_rosbag_recording()

    def shake(self):

        print("\nShaking...")
        behavior = self.robot + "_5-shake"

        self.start_rosbag_recording(behavior)

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
        behavior = self.robot + "_6-lower"

        # point = [-0.48, -0.25, 2.23, -3.08, -4.00, 2.24]
        point = [-0.63, -0.49, 2.29, -2.82, 2.18, 3.91]
        # point = [-0.48, -0.30, 2.24, -3.08, 2.28, 2.23]
        self.arm_group.set_joint_value_target(point)
        self.arm_group.go(wait=True)

        self.start_rosbag_recording(behavior)
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
        behavior = self.robot + "_7-drop"

        self.start_rosbag_recording(behavior)

        time.sleep(3.0)
        self.open_gripper()
        time.sleep(3.0)

        stop_rosbag_recording()

    def push(self, push_type):

        print("\nPushing...")

        if push_type == "slow":
            behavior = self.robot + "_8-push-slow"
        else:
            behavior = self.robot + "_8-push-fast"

        msg = GripperCmd(position=0.02, speed=1, force=100.0)
        rospy.sleep(1)
        self.gripper_pub.publish(msg)
        rospy.sleep(1)

        point = [-0.84, -1.19, 2.19, -1.91, 2.07, 2.59]
        self.arm_group.set_joint_value_target(point)
        self.arm_group.go(wait=True)

        self.start_rosbag_recording(behavior)
        time.sleep(2.0)

        if push_type == "slow":
            point = [-0.99, -0.83, 2.01, -2.04, 1.94, 2.77]
            self.arm_group.set_joint_value_target(point)
            self.arm_group.go(wait=True)
            time.sleep(2.0)
        else:
            point = [-0.99, -0.83, 2.01, -2.04, 1.94, 2.77]
            self.arm_client.add_point(point, 0.3)
            self.arm_client.start()
            self.arm_client.wait(0)
            self.arm_client.clear()

        stop_rosbag_recording()


if __name__ == "__main__":
    """
    This script performs look, grasp, pick, hold, shake, lower, drop, push behaviors.
    And saves sensory data:
        /joint_states, /wrench, /gripper/joint_states, /camera/rgb/image_raw, /camera/depth/image_raw, /audio
        /camera/depth_registered/points
    """

    arg_fmt = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=arg_fmt)
    required = parser.add_argument_group('required arguments')
    required.add_argument('-pt', '--push_type', required=False, choices=['slow', 'fast'],
                          help='Type of Push: Slow or fast')
    args = parser.parse_args(rospy.myargv()[1:])

    robot_name = 'ur5'
    colors = ['white', 'red', 'blue', 'green', 'yellow']
    weight_in_grams = [0, 50, 100, 150]
    contents = ['rice', 'pasta', 'nutsandbolts', 'marbles', 'dices', 'buttons']

    color = get_label_from_user(colors)
    weight = get_label_from_user(weight_in_grams)
    if weight == 0:
        weight = 22
        content = 'empty'
    else:
        content = get_label_from_user(contents)
    object_name = '-'.join([color, content, str(weight)+'g'])

    sensor_data_path = r"/media/gyan/Seagate Expansion Drive/UR5_Dataset/1_Raw/"

    trial_no = find_trial_no(object_name, sensor_data_path, robot_name)

    ans = ""
    while ans != 'y':
        try:
            ans = raw_input("Is it trial " + str(trial_no) + " of " + object_name + "? (Enter 'y' or the trial no.): ")
            ans = ans.lower()
            if ans != 'y':
                trial_no = int(ans)
        except ValueError:
            print("Sorry, I didn't understand that\n")
            continue
        else:
            break

    print("Recording trial {} of object {}".format(trial_no, object_name))

    sensor_data_path += os.sep + robot_name + "_" + object_name + os.sep + "trial-" + str(trial_no) + "_" + \
                        datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    b = Behaviors(sensor_data_path, robot_name)

    if args.push_type:
        b.push(args.push_type)
    else:
        b.look()
        b.grasp()
        b.pick()
        b.hold()
        b.shake()
        b.lower()
        b.drop()

    b.initialise_robot()
    time.sleep(2.0)

    ans = ""
    while ans not in ['y', 'n']:
        try:
            ans = raw_input("Do you want to save the data of trail " + str(trial_no) + " of " + object_name + "? (y/n): ")
        except ValueError:
            print("Sorry, I didn't understand that\n")
            continue
        else:
            ans = ans.lower()
            if ans not in ['y', 'n']:
                print("Answer you entered is not valid\n")
            else:
                if ans == 'n':
                    print("DELETING: ", sensor_data_path)
                    shutil.rmtree(sensor_data_path)
                break

    # Wait until bag files are written
    while True:
        active_bag_found = False
        for root, subdirs, files in os.walk(sensor_data_path):
            for filename in files:
                filename, fileext = os.path.splitext(filename)

                if fileext == '.active':
                    print("Active bag file: ", filename)
                    time.sleep(5)
                    active_bag_found = True
                    break

        if not active_bag_found:
            break

    print("Exiting - Joint Trajectory Action Complete!")

    """
    TODO:
    
    """
