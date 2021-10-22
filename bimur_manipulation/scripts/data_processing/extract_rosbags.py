#!/usr/bin/env python

# Gyan Tatiya

import json
import os
import subprocess
import wave

import cv2
from cv_bridge import CvBridge

import rosbag
from rospy_message_converter import message_converter as mc


def extract_img_from_bag(bag_, topic_, output_dir_):
    bridge = CvBridge()
    count = 0
    for topic__, msg, t in bag_.read_messages(topics=[topic_]):
        timestr = "%.9f" % msg.header.stamp.to_sec()
        # print("timestr: ", count, timestr)

        if 'depth' in topic_:
            cv_img = bridge.imgmsg_to_cv2(msg, msg.encoding)
        else:
            cv_img = bridge.imgmsg_to_cv2(msg, 'bgr8')

        cv2.imwrite(os.path.join(output_dir_, str(timestr) + '.png'), cv_img)

        count += 1


def extract_json_from_bag(bag_, topic_, output_dir_, topic_name_):
    sensor_msgs = []
    for topic__, msg, t in bag_.read_messages(topics=[topic_]):
        # print("msg: ", msg)
        sensor_msgs.append(mc.convert_ros_message_to_dictionary(msg))

    with open(output_dir_ + os.sep + topic_name_ + '.json', 'wb') as f:
        json.dump(sensor_msgs, f, indent=4)


def extract_audio_from_bag(bag_, topic_, output_dir_, topic_name_):
    sensor_msgs = []
    for topic__, msg, t in bag_.read_messages(topics=[topic_]):
        sensor_msgs.append(msg.data)

    # format = 8  # pyaudio.paInt16
    channels = 2  # 1, 2
    rate = 44100  # 16000, 44100

    wf = wave.open(output_dir_ + os.sep + topic_name_ + '.wav', 'wb')
    wf.setnchannels(channels)
    # wf.setsampwidth(self.p.get_sample_size(FORMAT))
    wf.setsampwidth(2L)
    wf.setframerate(rate)
    wf.writeframes(b''.join(sensor_msgs))
    wf.close()


def extract_pcd_from_bag(bag_file_, topic_, output_dir_):
    bag_to_pcd_cmd = "rosrun pcl_ros bag_to_pcd '" + bag_file_ + "' " + topic_ + " '" + output_dir_ + "'"
    print("bag_to_pcd_cmd: ", bag_to_pcd_cmd)
    p = subprocess.Popen(bag_to_pcd_cmd, stdin=subprocess.PIPE, shell=True)
    p.wait()


def main():
    """Extract files from rosbag
    """

    sensor_data_path = r"/media/gyan/My Passport/UR5_Dataset_Temp/"
    output_path = r"/media/gyan/My Passport/UR5_Dataset_Temp_Extrated/"

    camera_topics_to_make_video = ['/camera/rgb/image_raw']

    for root, subdirs, files in os.walk(sensor_data_path):
        for filename in files:
            print(filename)
            filename, fileext = os.path.splitext(filename)

            if fileext != '.bag':
                continue

            extracted_bag_path = root + os.sep + filename
            print("extracted_bag_path: ", extracted_bag_path)

            has_audio = False
            topics_has_video = {}

            bag = rosbag.Bag(extracted_bag_path + '.bag', "r")
            print("bag: ", bag)

            bag_info = bag.get_type_and_topic_info()
            print("len(bag_info): ", len(bag_info))
            print("type(bag_info): ", type(bag_info))
            print("bag_info: ", bag_info)
            print("")
            print("len(bag_info.topics): ", len(bag_info.topics))
            print("bag_info.topics: ", bag_info.topics)
            print("")
            print("len(bag_info.msg_types): ", len(bag_info.msg_types))
            print("bag_info.msg_types: ", bag_info.msg_types)
            print("=" * 100)

            for topic in bag_info.topics:
                print("topic: ", topic)
                print("msg_type: ", bag_info.topics[topic].msg_type)
                print("message_count: ", bag_info.topics[topic].message_count)
                print("frequency: ", bag_info.topics[topic].frequency)
                print("")

                topic_name = '_'.join(topic[1:].split('/'))
                output_dir = os.sep.join([output_path] + extracted_bag_path.split(os.sep)[-3:] + [topic_name])
                print("output_dir: ", output_dir)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                if topic == '/cameras/left_hand_camera/image' or topic == '/camera/rgb/image_raw' \
                        or topic == '/camera/depth/image_raw':
                    extract_img_from_bag(bag, topic, output_dir)
                    topics_has_video[topic] = True
                elif topic == '/joint_states' or topic == '/gripper/joint_states' or topic == '/wrench':
                    extract_json_from_bag(bag, topic, output_dir, topic_name)
                elif topic == '/audio':
                    extract_audio_from_bag(bag, topic, output_dir, topic_name)
                    has_audio = True
                elif topic == '/camera/depth_registered/points':
                    extract_pcd_from_bag(extracted_bag_path + '.bag', topic, output_dir)
                else:
                    print("{} cannot be extracted".format(topic))
                    exit()

                print("=" * 50)

            if has_audio:
                for video_topic in camera_topics_to_make_video:
                    if video_topic in topics_has_video and topics_has_video[video_topic]:
                        frequency = bag_info.topics[video_topic].frequency
                        video_topic_name = '_'.join(video_topic[1:].split('/'))
                        video_output_dir = os.sep.join([output_path] + extracted_bag_path.split(os.sep)[-3:] + [video_topic_name])

                        audio_topic = '/audio'
                        audio_topic_name = '_'.join(audio_topic[1:].split('/'))
                        audio_output_dir = os.sep.join([output_path] + extracted_bag_path.split(os.sep)[-3:] + [audio_topic_name])

                        cmd = "ffmpeg -r " + str(frequency) + " -pattern_type glob -i '" + video_output_dir + os.sep + \
                              "*.png' -i '" + audio_output_dir + os.sep + audio_topic_name + ".wav'" + \
                              " -strict experimental -async 1 '" + video_output_dir + "_video.mp4' -y"
                        print("cmd: ", cmd)
                        p = subprocess.Popen(cmd, stdin=subprocess.PIPE, shell=True)

            bag.close()


if __name__ == '__main__':
    main()

    """
    TODO:
    Save logs
    
    For Stage 1, make a config.py for objects_id, topics
    """
