# Author: Gyan Tatiya

import json
import os
import pickle

import cv2
import librosa
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize


def read_images(root_, files_):
    data_ = []

    for filename_ in sorted(files_):
        image = cv2.imread(root_ + os.sep + filename_, cv2.IMREAD_UNCHANGED)
        data_.append(image)

    return np.array(data_)


def discretize_data(data_, discretize_temporal_bins_):
    frames = data_.shape[0]
    dimension = data_.shape[1]
    # Fix if number of frames is less than temporal_bins
    if frames < discretize_temporal_bins_:
        print(frames, " is less than " + str(discretize_temporal_bins_) + " frames")
        data_ = resize(data_, (discretize_temporal_bins_, dimension))
        frames = data_.shape[0]
    size = frames // discretize_temporal_bins_

    discretized_data = []
    for a_bin in range(discretize_temporal_bins_):
        value = np.mean(data_[size * a_bin:size * (a_bin + 1)], axis=0)
        discretized_data.append(value)

    return np.array(discretized_data)


def discretize_data_v2(data_, discretize_temporal_bins_):
    frames = data_.shape[0]
    dimension = data_.shape[1]
    # Fix if number of frames is less than temporal_bins
    if frames < discretize_temporal_bins_:
        print(frames, " is less than " + str(discretize_temporal_bins_) + " frames")
        data_ = resize(data_, (discretize_temporal_bins_, dimension))
        frames = data_.shape[0]
    frames_size = frames // discretize_temporal_bins_
    dimension_size = dimension // discretize_temporal_bins_

    discretized_data = []
    for a_bin in range(discretize_temporal_bins_):
        discretized_data2 = []
        for a_bin2 in range(discretize_temporal_bins_):
            value = np.mean(data_[frames_size * a_bin:frames_size * (a_bin + 1),
                            dimension_size * a_bin2:dimension_size * (a_bin2 + 1)])
            discretized_data2.append(value)
        discretized_data.append(discretized_data2)

    return np.array(discretized_data)


def read_joint_states(data_, joint_state_, joints_idx_):
    effort_data = []
    for i in range(len(data_)):
        try:
            effort_data.append(list(np.array(data_[i][joint_state_])[joints_idx_]))
        except:
            pass

    effort_data = np.array(effort_data)

    return effort_data


def add_data(discretize_temporal_bins_, behavior_, modality_, trial_, object_name_, data_, behavior_data_,
             dataset_metadata_):

    if discretize_temporal_bins_:
        if 'audio' == modality:
            data_ = discretize_data_v2(data_, discretize_temporal_bins_)
        else:
            data_ = discretize_data(data_, discretize_temporal_bins_)
        behavior_data_[object_name_][trial_].setdefault(modality_ + '-discretized', data_)
        dataset_metadata_[behavior_][object_name_][trial_].setdefault(modality_ + '-discretized', data_.shape)
    else:
        behavior_data_[object_name_][trial_].setdefault(modality_, data_)
        dataset_metadata_[behavior_][object_name_][trial_].setdefault(modality_, data_.shape[1:])

    return behavior_data_, dataset_metadata_


def dump_data(data_, dataset_path_, behavior_, object_name_, trial_, modality_):

    dataset_path_ = os.sep.join([dataset_path_, behavior_, object_name_, trial_])
    os.makedirs(dataset_path_, exist_ok=True)
    db_file_name_ = dataset_path_ + os.sep + modality_ + ".bin"
    output_file_ = open(db_file_name_, "wb")
    pickle.dump(data_, output_file_)
    output_file_.close()


if __name__ == "__main__":

    sensor_data_path = r"/media/gyan/My Passport/UR5_Dataset/2_Extracted"
    dataset_path = r"/media/gyan/My Passport/UR5_Dataset/3_Binary"

    file_formats = ['.wav', '.png', '.json']

    behaviors = ['look', 'grasp', 'pick', 'hold', 'shake', 'lower', 'drop', 'push']  # ['shake']
    discretize_temporal_bins = 10  # 0 for not discretizing

    modalities = {'camera_depth_image_raw', 'camera_rgb_image_raw', 'effort', 'position', 'velocity',
                  'gripper_joint_states', 'torque', 'force', 'audio'}
    if discretize_temporal_bins:
        modalities_discretized = {'camera_depth_image_raw', 'camera_rgb_image_raw'}
        for m in modalities:
            if m not in modalities_discretized:
                modalities_discretized.add(m + '-discretized')
        modalities = modalities_discretized
    print("modalities: ", modalities)

    ds_metadata_full_filename = "dataset_metadata_full_discretized.bin" if discretize_temporal_bins else "dataset_metadata_full.bin"
    ds_metadata_full_path = dataset_path + os.sep + ds_metadata_full_filename
    if os.path.exists(ds_metadata_full_path):
        print("Loading dataset_metadata")
        bin_file = open(ds_metadata_full_path, "rb")
        dataset_metadata = pickle.load(bin_file)
        bin_file.close()
    else:
        dataset_metadata = {}

    for behavior in behaviors:
        behavior_data = {}
        dataset_metadata.setdefault(behavior, {})
        for root, subdirs, files in os.walk(sensor_data_path):
            print("root: ", root)
            print("subdirs: ", subdirs)
            print("files: ", len(files), files[:5])
            for filename in files:
                print("filename ", filename)
                filename, fileext = os.path.splitext(filename)

                if fileext in file_formats:
                    root_list = root.split(os.sep)
                    curr_behavior = root_list[-2].split('_')[1].split('-')[1]
                    print("curr_behavior: ", curr_behavior)
                    if curr_behavior == behavior:
                        modality = root_list[-1]
                        trial = root_list[-3].split('_')[0]  # .split('-')[1]
                        object_name = root_list[-4].split('_')[1]

                        # Skipping processed data
                        dataset_temp_path = os.sep.join([dataset_path, behavior, object_name, trial])
                        modality_temp = modality
                        if discretize_temporal_bins and fileext != '.png':
                            modality_temp = modality + '-discretized'
                        db_temp_file_name = dataset_temp_path + os.sep + modality_temp + ".bin"
                        if os.path.exists(db_temp_file_name):
                            print(db_temp_file_name + " already exists, so skipping ...")

                            # Restoring the metadata for vision data from old processing
                            ds_metadata_full_filename = "dataset_metadata_full.bin" if discretize_temporal_bins else "dataset_metadata_full_discretized.bin"
                            ds_metadata_full_temp_path = dataset_path + os.sep + ds_metadata_full_filename
                            if os.path.exists(ds_metadata_full_temp_path):
                                bin_file = open(ds_metadata_full_temp_path, "rb")
                                dataset_metadata_temp = pickle.load(bin_file)
                                bin_file.close()

                                for mod in ['camera_depth_image_raw', 'camera_rgb_image_raw']:
                                    dataset_metadata[behavior].setdefault(object_name, {}).setdefault(trial, {})
                                    dataset_metadata[behavior][object_name][trial][mod] = dataset_metadata_temp[behavior][object_name][trial][mod]

                            break

                        # if trial not in ['trial-0'] or object_name not in ['red-pasta-150g'] or modality not in ['joint_states']:
                        #     continue

                        behavior_data.setdefault(object_name, {}).setdefault(trial, {})
                        dataset_metadata[behavior].setdefault(object_name, {}).setdefault(trial, {})

                        print("modality: ", modality)
                        print("trial: ", trial)
                        print("object_name: ", object_name)

                        if fileext == '.png':
                            data = read_images(root, files)
                            print("data: ", len(data))
                            behavior_data[object_name][trial].setdefault(modality, data)
                            dataset_metadata[behavior][object_name][trial].setdefault(modality, data.shape[1:])
                            # print("behavior_data: ", behavior_data)
                        elif fileext == '.wav':
                            audio_time_series, sampling_rate = librosa.load(root + os.sep + files[0], sr=44100)
                            # print("audio_time_series: ", audio_time_series.shape)
                            # print("sampling_rate: ", sampling_rate)

                            audio_Length = len(audio_time_series) / sampling_rate
                            # print("audio_Length: ", audio_Length)

                            melspec = librosa.feature.melspectrogram(audio_time_series, sr=sampling_rate, n_fft=1024,
                                                                     hop_length=512, n_mels=60)
                            # print("melspec: ", melspec.shape)
                            logspec = librosa.core.amplitude_to_db(melspec)
                            logspec = np.transpose(logspec)
                            # print("logspec: ", logspec.shape)

                            #############

                            # n_fft = 512
                            # hop_length = 160
                            # win_length = 400
                            # stft = np.abs(librosa.stft(audio_time_series, n_fft=n_fft, hop_length=hop_length, win_length=win_length))
                            # # stft = block_reduce(stft, block_size=(4, 4), func=np.mean)
                            # print("stft: ", stft.shape)

                            behavior_data, dataset_metadata = add_data(discretize_temporal_bins, behavior, modality,
                                                                       trial, object_name, logspec, behavior_data,
                                                                       dataset_metadata)
                        elif fileext == '.json':
                            f = open(root + os.sep + files[0])
                            data = json.load(f)
                            print("data: ", len(data))
                            print("data: ", data[0].keys())
                            print("data[0]: ", data[0])
                            # print("data: ", data[0]['position'])
                            print("")
                            print("data[1]: ", data[1])
                            # print("data: ", data[1]['position'])

                            temporal_bins = 10
                            if modality == 'joint_states':
                                joints_idx = []
                                i = 0
                                while not joints_idx:
                                    for j in range(len(data[i]['name'])):
                                        if 'gripper' not in data[i]['name'][j]:
                                            joints_idx.append(j)
                                    if not joints_idx:
                                        i += 1
                                assert joints_idx, "No joints found"
                                print("joints_idx: ", joints_idx)

                                for joint_state in ['effort', 'position', 'velocity']:
                                    print("joint_state: ", joint_state)
                                    joint_state_data = read_joint_states(data, joint_state, joints_idx)
                                    print("joint_state_data: ", joint_state_data.shape)

                                    behavior_data, dataset_metadata = add_data(discretize_temporal_bins, behavior,
                                                                               joint_state, trial, object_name,
                                                                               joint_state_data, behavior_data,
                                                                               dataset_metadata)
                                    # print("behavior_data: ", behavior_data[object_name][trial][joint_state+'-discretized'].shape)
                            elif modality == 'gripper_joint_states':
                                gripper_state_data = []
                                for i in range(len(data)):
                                    gripper_state_data.append([data[i]['velocity'][0], data[i]['position'][0]])
                                gripper_state_data = np.array(gripper_state_data)
                                print("gripper_state_data: ", gripper_state_data.shape)

                                behavior_data, dataset_metadata = add_data(discretize_temporal_bins, behavior, modality,
                                                                           trial, object_name, gripper_state_data,
                                                                           behavior_data, dataset_metadata)
                            elif modality == 'wrench':

                                for wrench in ['torque', 'force']:

                                    wrench_data = []
                                    for i in range(len(data)):
                                        wrench_data.append([data[i]['wrench'][wrench]['x'],
                                                            data[i]['wrench'][wrench]['y'],
                                                            data[i]['wrench'][wrench]['z']])
                                    wrench_data = np.array(wrench_data)
                                    print("wrench_data: ", wrench_data.shape)

                                    behavior_data, dataset_metadata = add_data(discretize_temporal_bins, behavior,
                                                                               wrench, trial, object_name, wrench_data,
                                                                               behavior_data, dataset_metadata)
                            else:
                                assert False, "Modality (" + modality + ") not supported"
                        else:
                            assert False, "File extension (" + fileext + ") not supported"

                        print("behavior_data[object_name][trial]: ", set(behavior_data[object_name][trial].keys()))

                        for modality in behavior_data[object_name][trial].keys():
                            print("Saving data: ", behavior, object_name, trial, modality)
                            dump_data(behavior_data[object_name][trial][modality], dataset_path, behavior, object_name,
                                      trial, modality)
                        # clean after dump
                        modalities_dump = set(behavior_data[object_name][trial].keys())
                        for modality in modalities_dump:
                            del behavior_data[object_name][trial][modality]

                        output_file = open(ds_metadata_full_path, "wb")
                        pickle.dump(dataset_metadata, output_file)
                        output_file.close()

                        print("")
                        break
                    else:
                        break
                else:
                    break

    metadata_new = {}
    for behavior in dataset_metadata:
        print("behavior: ", behavior)
        metadata_new.setdefault(behavior, {})
        metadata_new[behavior]['objects'] = set(dataset_metadata[behavior].keys())

        for object_name in dataset_metadata[behavior]:
            print("object_name: ", object_name)
            metadata_new[behavior]['trials'] = set(dataset_metadata[behavior][object_name].keys())

            for trial in dataset_metadata[behavior][object_name]:
                print("trial: ", trial)

                for modality in dataset_metadata[behavior][object_name][trial]:
                    print("modality: ", modality)
                    print("shape: ", dataset_metadata[behavior][object_name][trial][modality])

                    metadata_new[behavior].setdefault('modalities', {})
                    metadata_new[behavior]['modalities'].setdefault(modality,
                                                                    dataset_metadata[behavior][object_name][trial][modality])
                    if metadata_new[behavior]['modalities'][modality]:
                        assert metadata_new[behavior]['modalities'][modality] == dataset_metadata[behavior][object_name][trial][modality],\
                            "Size mismatch: {}, {}, {}, {}: {} != {}".format(behavior, object_name, trial, modality,
                                                                             metadata_new[behavior]['modalities'][modality],
                                                                             dataset_metadata[behavior][object_name][trial][modality])

        print("metadata_new: ", metadata_new)

    ds_metadata_filename = "dataset_metadata_discretized.bin" if discretize_temporal_bins else "dataset_metadata.bin"
    ds_metadata_path = dataset_path + os.sep + ds_metadata_filename
    output_file = open(ds_metadata_path, "wb")
    pickle.dump(metadata_new, output_file)
    output_file.close()

    print("=============================:)=============================")
