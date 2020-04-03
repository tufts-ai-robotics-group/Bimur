#!/usr/bin/env python
import sys
import time
import unittest

import rospy
import actionlib
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectoryPoint

PKG = 'ur_rtde_driver'
NAME = 'trajectory_test'


class TrajectoryTest(unittest.TestCase):
    def __init__(self, *args):
        super(TrajectoryTest, self).__init__(*args)
        rospy.init_node('trajectory_testing_client')
        self.client = actionlib.SimpleActionClient(
            '/scaled_pos_traj_controller/follow_joint_trajectory', FollowJointTrajectoryAction)
        timeout = rospy.Duration(10)
        try:
            self.client.wait_for_server(timeout)
        except rospy.exceptions.ROSException as err:
            self.fail(
                "Could not reach controller action. Make sure that the driver is actually running."
                " Msg: {}".format(err))

        # rospy.sleep(5)

    def test_trajectory(self):
        """Test robot movement"""
        goal = FollowJointTrajectoryGoal()

        goal.trajectory.joint_names = ["elbow_joint", "shoulder_lift_joint", "shoulder_pan_joint",
                                       "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
        # position_list = [[0.0 for i in range(6)]]
        position_list = [[2.10, -1.09, -0.2, -3.12, 2.65, 0.00]]
        position_list.append([2.17, -1.7, -0.45, -1.55, 2.3, 2.25])
        position_list.append([2.24, -1.35, -0.8, -1.8, 2.12, 2.5])
        duration_list = [12.0, 20.0, 30.0]

        for i, position in enumerate(position_list):
            print("position: ", position)
            point = JointTrajectoryPoint()
            point.positions = position
            point.time_from_start = rospy.Duration(duration_list[i])
            goal.trajectory.points.append(point)

        self.client.send_goal(goal)
        self.client.wait_for_result()

        self.assertEqual(self.client.get_result().error_code, 0)


if __name__ == '__main__':
    import rostest
    rostest.run(PKG, NAME, TrajectoryTest, sys.argv)
