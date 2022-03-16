# Author: Gyan Tatiya

import os
import pickle

import cv2
import matplotlib.pyplot as plt
import numpy as np


def print_image(image, modality_, output_dir_):

    if 'audio' in modality_:
        plt.imshow(np.flipud(image.T))
    else:
        plt.imshow(image)
    # plt.imshow(image, cmap=plt.cm.gray)
    # plt.imshow(image, origin="lower", cmap="jet")
    # plt.imshow(image, origin="lower", aspect="auto", cmap="jet", interpolation="none")
    plt.title(modality_)
    plt.colorbar()
    plt.savefig(output_dir_ + os.sep + modality_ + ".png", bbox_inches='tight', dpi=100)
    # plt.show()
    plt.close()


def plot_an_example_discretized(example_, modality_, output_dir_):

    plt.title(modality_, fontsize=16)
    plt.xlabel('Temporal Bins', fontsize=16)
    plt.ylabel('Joints', fontsize=16)
    if 'audio' in modality_:
        plt.imshow(np.flipud(example_.T), cmap="gist_ncar")
    else:
        plt.imshow(example_.T, cmap="gist_ncar")
    x_values, y_values = example_.shape
    ax = plt.gca()
    ax.set_xticks(np.arange(0, x_values, 1))
    ax.set_yticks(np.arange(0, y_values, 1))
    ax.set_xticklabels(np.arange(1, x_values + 1, 1))
    ax.set_yticklabels(np.arange(1, y_values + 1, 1))
    plt.colorbar(orientation='vertical')
    plt.savefig(output_dir_ + os.sep + modality_ + ".png", bbox_inches='tight', dpi=100)
    # plt.show()
    plt.close()


def plot_an_example_for_xyz(example_, modality_, output_dir_):

    axis = ['x', 'y', 'z']
    for i in range(len(axis)):
        plt.plot(example_[:, i], label=axis[i])

    plt.title(modality_, fontsize=16)
    plt.xlabel('Samples', fontsize=16)
    plt.ylabel(modality, fontsize=16)
    plt.legend(title='Axis', loc='upper right')
    plt.savefig(output_dir_ + os.sep + modality_ + ".png", bbox_inches='tight', dpi=100)
    # plt.show()
    plt.close()


def plot_an_example(example_, modality_, output_dir_):

    joints = example_.shape[1]
    for i in range(joints):
        plt.plot(example_[:, i], label=str(i+1))

    plt.title(modality_, fontsize=16)
    plt.xlabel('Samples', fontsize=16)
    plt.ylabel(modality, fontsize=16)
    plt.legend(title='Joints', loc='upper right')
    plt.savefig(output_dir_ + os.sep + modality_ + ".png", bbox_inches='tight', dpi=100)
    # plt.show()
    plt.close()


if __name__ == "__main__":

    dataset_path = r"/media/gyan/My Passport/UR5_Dataset/3_Binary"
    output_path = r"/media/gyan/My Passport/UR5_Dataset/4_Plots"

    for root, subdirs, files in os.walk(dataset_path):
        for filename in files:
            print(filename)
            filename, fileext = os.path.splitext(filename)

            if fileext != '.bin' or 'metadata' in filename:
                continue

            trial_data_path = root + os.sep + filename
            print("trial_data_path: ", trial_data_path)

            behavior = trial_data_path.split(os.sep)[-4]
            object_name = trial_data_path.split(os.sep)[-3]
            trial = trial_data_path.split(os.sep)[-2]
            modality = trial_data_path.split(os.sep)[-1]

            print("behavior: ", behavior)
            print("object_name: ", object_name)
            print("trial: ", trial)
            print("modality: ", modality)

            data_file_path = trial_data_path + '.bin'
            bin_file = open(data_file_path, "rb")
            example = pickle.load(bin_file)
            bin_file.close()

            print("example: ", example.shape)

            output_dir = os.path.join(output_path, behavior, object_name, trial)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            if modality == 'camera_rgb_image_raw':
                output_dir = output_dir + os.sep + modality
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                for i in range(len(example)):
                    # Quality for JPEG encoding in range 1-100
                    cv2.imwrite(output_dir + os.sep + str(i) + ".jpg", example[i], [cv2.IMWRITE_JPEG_QUALITY, 80])

            elif modality in ['effort-discretized', 'position-discretized', 'velocity-discretized']:
                plot_an_example_discretized(example, modality, output_dir)
            elif modality in ['effort', 'position', 'velocity']:
                plot_an_example(example, modality, output_dir)

            elif modality == 'gripper_joint_states-discretized':
                plot_an_example_discretized(example, modality, output_dir)
            elif modality == 'gripper_joint_states':
                plot_an_example(example, modality, output_dir)

            elif modality in ['torque-discretized', 'force-discretized']:
                plot_an_example_discretized(example, modality, output_dir)
            elif modality in ['torque', 'force']:
                plot_an_example_for_xyz(example, modality, output_dir)

            elif modality in ['audio-discretized']:
                plot_an_example_discretized(example, modality, output_dir)
            elif modality in ['audio']:
                print_image(example, modality, output_dir)

            print("")

    """
    TODO:
    
    """
