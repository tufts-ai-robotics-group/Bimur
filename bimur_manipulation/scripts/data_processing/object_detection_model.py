# Author: Gyan Tatiya

import csv
import os
import pickle

import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB


def split_train_test(n_folds, trials_per_object):

    test_size = trials_per_object // n_folds
    tt_splits = {}

    for a_fold in range(n_folds):

        train_index = []
        test_index = np.arange(test_size * a_fold, test_size * (a_fold + 1))

        if test_size * a_fold > 0:
            train_index.extend(np.arange(0, test_size * a_fold))
        if test_size * (a_fold + 1) - 1 < trials_per_object - 1:
            train_index.extend(np.arange(test_size * (a_fold + 1), trials_per_object))

        tt_splits.setdefault("fold_" + str(a_fold), {}).setdefault("train", []).extend(train_index)
        tt_splits.setdefault("fold_" + str(a_fold), {}).setdefault("test", []).extend(test_index)

    return tt_splits


def get_split_data(trials, objects_labels_, path, objects, behavior_, modality_):

    x_split = []
    y_split = []
    for object_name_ in sorted(objects):
        for trial_num in sorted(trials):
            data_path = os.sep.join([path, behavior_, object_name_, "trial-" + str(trial_num), modality_ + ".bin"])
            bin_file_ = open(data_path, "rb")
            example = pickle.load(bin_file_)
            bin_file_.close()

            if DETECTION_TASK == 'objects':
                x_split.append(example.flatten())
                y_split.append(objects_labels_[object_name_])
            elif DETECTION_TASK == 'weights':
                for w in [22, 50, 100, 150]:
                    w = str(w) + 'g'
                    if w == object_name_.split('-')[2]:
                        x_split.append(example.flatten())
                        y_split.append(objects_labels_[w])
                        break
            elif DETECTION_TASK == 'contents':
                for c in ['rice', 'pasta', 'nutsandbolts', 'marbles', 'dices', 'buttons']:
                    if c == object_name_.split('-')[1]:
                        x_split.append(example.flatten())
                        y_split.append(objects_labels_[c])
                        break
            elif DETECTION_TASK == 'colors':
                for c in ['white', 'red', 'blue', 'green', 'yellow']:
                    if c == object_name_.split('-')[0]:
                        # extracting histogram features from the last image
                        example = cv2.calcHist([example[-1]], [0, 1, 2], None, [4, 4, 4], [0, 256, 0, 256, 0, 256])
                        # print("3D histogram shape: {}, with {} values".format(example.shape, example.flatten().shape[0]))
                        x_split.append(example.flatten())
                        y_split.append(objects_labels_[c])
                        break

    return x_split, y_split


def classifier(my_classifier, x_train_temp, x_test_temp, y_train_temp, y_test_temp):
    """
    Train a classifier on test data and return accuracy and prediction on test data
    :param my_classifier:
    :param x_train_temp:
    :param x_test_temp:
    :param y_train_temp:
    :param y_test_temp:
    :return: accuracy, prediction
    """

    # Fit the model on the training data.
    my_classifier.fit(x_train_temp, y_train_temp)

    # See how the model performs on the test data.
    accuracy = my_classifier.score(x_test_temp, y_test_temp)
    prediction = my_classifier.predict(x_test_temp)
    probability = my_classifier.predict_proba(x_test_temp)

    return accuracy, prediction, probability


def combine_probability(proba_acc_list_, y_test_, acc=None):

    # For each classifier, combine weighted probability based on its accuracy score
    proba_list = []
    for proba_acc in proba_acc_list_:
        y_proba = proba_acc['proba']
        if acc and proba_acc[acc] > 0:
            # Multiple the score by probability to combine each classifier's performance accordingly
            y_proba = y_proba * proba_acc[acc]  # weighted probability
        proba_list.append(y_proba)

    # Combine weighted probability of all classifiers
    y_proba_norm = np.zeros(len(proba_list[0][0]))
    for proba in proba_list:
        y_proba_norm = y_proba_norm + proba

    # Normalizing probability
    y_proba_norm_sum = np.sum(y_proba_norm, axis=1)  # sum of weighted probability
    y_proba_norm_sum = np.repeat(y_proba_norm_sum, len(proba_list[0][0]), axis=0).reshape(y_proba_norm.shape)
    y_proba_norm = y_proba_norm / y_proba_norm_sum

    y_proba_pred = np.argmax(y_proba_norm, axis=1)
    y_prob_acc = np.mean(y_test_ == y_proba_pred)

    return y_proba_norm, y_prob_acc


if __name__ == "__main__":

    dataset_path = r"/media/wildog2-1/Samsung 870 QVO/UR5_Dataset/3_Binary"

    db_file_name = dataset_path + os.sep + "dataset_metadata_discretized.bin"
    bin_file = open(db_file_name, "rb")
    metadata = pickle.load(bin_file)
    bin_file.close()

    # print("metadata: ", metadata)

    num_of_test_examples = 1

    DETECTION_TASK = 'colors'  # objects, weights, contents, colors

    # CLF = SVC(gamma='auto', kernel='rbf', probability=True)
    # CLF_NAME = "SVM-RBF"

    CLF = SVC(gamma='auto', kernel='linear', probability=True)
    CLF_NAME = "SVM-LIN"

    # CLF = KNeighborsClassifier(n_neighbors=1)
    # CLF_NAME = "KNN"

    # CLF = DecisionTreeClassifier()
    # CLF_NAME = "DT"

    # CLF = RandomForestClassifier()
    # CLF_NAME = "RF"

    # CLF = AdaBoostClassifier()
    # CLF_NAME = "AB"

    # CLF = GaussianNB()
    # CLF_NAME = "GN"

    results_path = 'results'

    folds = len(metadata['grasp']['trials']) // num_of_test_examples
    train_test_splits = split_train_test(folds, len(metadata['grasp']['trials']))
    print("train_test_splits: ", train_test_splits)

    folds_behaviors_modalities_proba_score = {}
    for fold in sorted(train_test_splits):
        print(fold)
        folds_behaviors_modalities_proba_score.setdefault(fold, {})
        for behavior in metadata:
            print("behavior: ", behavior)
            print("objects: ", len(metadata[behavior]['objects']), metadata[behavior]['objects'])
            print("trials: ", len(metadata[behavior]['trials']), metadata[behavior]['trials'])
            print("modality: ", len(metadata[behavior]['modalities']), metadata[behavior]['modalities'])

            if (DETECTION_TASK != 'colors' and behavior == 'look') or (DETECTION_TASK == 'colors' and behavior != 'look'):
                continue

            folds_behaviors_modalities_proba_score[fold].setdefault(behavior, {})

            objects_labels = {}
            if DETECTION_TASK == 'objects':
                for i, object_name in enumerate(sorted(metadata[behavior]['objects'])):
                    objects_labels[object_name] = i
            elif DETECTION_TASK == 'weights':
                for i, weight in enumerate(sorted([22, 50, 100, 150])):
                # for i, weight in enumerate(sorted([50, 150])):
                    objects_labels[str(weight) + 'g'] = i
            elif DETECTION_TASK == 'contents':
                for i, content in enumerate(sorted(['rice', 'pasta', 'nutsandbolts', 'marbles', 'dices', 'buttons'])):
                    objects_labels[content] = i
            elif DETECTION_TASK == 'colors':
                for i, content in enumerate(sorted(['white', 'red', 'blue', 'green', 'yellow'])):
                    objects_labels[content] = i
            print("objects_labels: ", objects_labels)

            # For each modality, combine weighted probability based on its accuracy score
            for modality in metadata[behavior]['modalities']:
                print("modality: ", modality)
                if modality in {'camera_depth_image_raw'}:
                    continue

                folds_behaviors_modalities_proba_score[fold][behavior].setdefault(modality, {})

                # Get train data
                X_train, y_train = get_split_data(train_test_splits[fold]['train'], objects_labels, dataset_path,
                                                  metadata[behavior]['objects'], behavior, modality)
                # print("X_train: ", len(X_train))
                # print("y_train: ", len(y_train), y_train)

                # Get test data
                X_test, y_test = get_split_data(train_test_splits[fold]['test'], objects_labels, dataset_path,
                                                metadata[behavior]['objects'], behavior, modality)
                # print("X_test: ", len(X_test))
                # print("y_test: ", len(y_test), y_test)

                # Train and Test
                y_acc, y_pred, y_proba = classifier(CLF, X_train, X_test, y_train, y_test)

                y_proba_pred = np.argmax(y_proba, axis=1)
                y_prob_acc = np.mean(y_test == y_proba_pred)
                print("y_prob_acc: ", y_prob_acc)

                folds_behaviors_modalities_proba_score[fold][behavior][modality]['proba'] = y_proba
                folds_behaviors_modalities_proba_score[fold][behavior][modality]['test_acc'] = y_prob_acc

                # For each behavior, get an accuracy score to combine weighted probability based on its accuracy score
                # Use only training data to get a score
                y_acc_train, y_pred_train, y_proba_train = classifier(CLF, X_train, X_train, y_train, y_train)
                y_proba_pred_train = np.argmax(y_proba_train, axis=1)
                y_prob_acc_train = np.mean(y_train == y_proba_pred_train)
                print("y_prob_acc_train: ", y_prob_acc_train)

                folds_behaviors_modalities_proba_score[fold][behavior][modality]['train_acc'] = y_prob_acc_train

            # For each modality, combine weighted probability based on its accuracy score
            proba_acc_list = []
            for modality in folds_behaviors_modalities_proba_score[fold][behavior]:
                proba_acc = {'proba': folds_behaviors_modalities_proba_score[fold][behavior][modality]['proba'],
                             'train_acc': folds_behaviors_modalities_proba_score[fold][behavior][modality]['train_acc'],
                             'test_acc': folds_behaviors_modalities_proba_score[fold][behavior][modality]['test_acc']}
                proba_acc_list.append(proba_acc)

            y_proba_norm, y_prob_acc = combine_probability(proba_acc_list, y_test)
            folds_behaviors_modalities_proba_score[fold][behavior].setdefault('all_modalities', {})
            folds_behaviors_modalities_proba_score[fold][behavior]['all_modalities']['proba'] = y_proba_norm
            folds_behaviors_modalities_proba_score[fold][behavior]['all_modalities']['test_acc'] = y_prob_acc

            y_proba_norm, y_prob_acc = combine_probability(proba_acc_list, y_test, 'train_acc')
            folds_behaviors_modalities_proba_score[fold][behavior].setdefault('all_modalities_train', {})
            folds_behaviors_modalities_proba_score[fold][behavior]['all_modalities_train']['proba'] = y_proba_norm
            folds_behaviors_modalities_proba_score[fold][behavior]['all_modalities_train']['test_acc'] = y_prob_acc

            y_proba_norm, y_prob_acc = combine_probability(proba_acc_list, y_test, 'test_acc')
            folds_behaviors_modalities_proba_score[fold][behavior].setdefault('all_modalities_test', {})
            folds_behaviors_modalities_proba_score[fold][behavior]['all_modalities_test']['proba'] = y_proba_norm
            folds_behaviors_modalities_proba_score[fold][behavior]['all_modalities_test']['test_acc'] = y_prob_acc

        # For each behavior and modality, combine weighted probability based on its accuracy score
        proba_acc_list = []
        for behavior in folds_behaviors_modalities_proba_score[fold]:
            for modality in folds_behaviors_modalities_proba_score[fold][behavior]:
                if not modality.startswith('all_modalities'):
                    proba_acc = {'proba': folds_behaviors_modalities_proba_score[fold][behavior][modality]['proba'],
                                 'train_acc': folds_behaviors_modalities_proba_score[fold][behavior][modality]['train_acc'],
                                 'test_acc': folds_behaviors_modalities_proba_score[fold][behavior][modality]['test_acc']}
                    proba_acc_list.append(proba_acc)

        y_proba_norm, y_prob_acc = combine_probability(proba_acc_list, y_test)
        folds_behaviors_modalities_proba_score[fold].setdefault('all_behaviors_modalities', {})
        folds_behaviors_modalities_proba_score[fold]['all_behaviors_modalities']['proba'] = y_proba_norm
        folds_behaviors_modalities_proba_score[fold]['all_behaviors_modalities']['test_acc'] = y_prob_acc

        y_proba_norm, y_prob_acc = combine_probability(proba_acc_list, y_test, 'train_acc')
        folds_behaviors_modalities_proba_score[fold].setdefault('all_behaviors_modalities_train', {})
        folds_behaviors_modalities_proba_score[fold]['all_behaviors_modalities_train']['proba'] = y_proba_norm
        folds_behaviors_modalities_proba_score[fold]['all_behaviors_modalities_train']['test_acc'] = y_prob_acc

        y_proba_norm, y_prob_acc = combine_probability(proba_acc_list, y_test, 'test_acc')
        folds_behaviors_modalities_proba_score[fold].setdefault('all_behaviors_modalities_test', {})
        folds_behaviors_modalities_proba_score[fold]['all_behaviors_modalities_test']['proba'] = y_proba_norm
        folds_behaviors_modalities_proba_score[fold]['all_behaviors_modalities_test']['test_acc'] = y_prob_acc

    behaviors_modalities_score = {}
    for fold in folds_behaviors_modalities_proba_score:
        for behavior in folds_behaviors_modalities_proba_score[fold]:
            if behavior.startswith('all_behaviors_modalities'):
                behaviors_modalities_score.setdefault(behavior, [])
                y_prob_acc = folds_behaviors_modalities_proba_score[fold][behavior]['test_acc']
                behaviors_modalities_score[behavior].append(y_prob_acc)
            else:
                behaviors_modalities_score.setdefault(behavior, {})
                for modality in folds_behaviors_modalities_proba_score[fold][behavior]:
                    behaviors_modalities_score[behavior].setdefault(modality, [])
                    y_prob_acc = folds_behaviors_modalities_proba_score[fold][behavior][modality]['test_acc']
                    behaviors_modalities_score[behavior][modality].append(y_prob_acc)

    for behavior in behaviors_modalities_score:
        if behavior.startswith('all_behaviors_modalities'):
            behaviors_modalities_score[behavior] = np.mean(behaviors_modalities_score[behavior])
        else:
            for modality in behaviors_modalities_score[behavior]:
                behaviors_modalities_score[behavior][modality] = np.mean(behaviors_modalities_score[behavior][modality])

    for behavior in behaviors_modalities_score:
        print(behavior, behaviors_modalities_score[behavior])

    row = ["behavior"]
    b = list(behaviors_modalities_score.keys())[0]
    for modality in behaviors_modalities_score[b]:
        if modality == 'audio-discretized':
            row.append('audio')
        elif modality == 'gripper_joint_states-discretized':
            row.append('gripper')
        elif modality == 'effort-discretized':
            row.append('effort')
        elif modality == 'position-discretized':
            row.append('position')
        elif modality == 'velocity-discretized':
            row.append('velocity')
        elif modality == 'torque-discretized':
            row.append('torque')
        elif modality == 'force-discretized':
            row.append('force')
        elif modality == 'camera_rgb_image_raw':
            row.append('camera_rgb')
        else:
            row.append(modality)

    df = pd.DataFrame(columns=row)
    for behavior in behaviors_modalities_score:

        if not behavior.startswith('all_behaviors_modalities'):
            row = {"behavior": behavior}
            for modality in behaviors_modalities_score[behavior]:
                if modality == 'audio-discretized':
                    row['audio'] = round(behaviors_modalities_score[behavior][modality] * 100, 2)
                elif modality == 'gripper_joint_states-discretized':
                    row['gripper'] = round(behaviors_modalities_score[behavior][modality] * 100, 2)
                elif modality == 'effort-discretized':
                    row['effort'] = round(behaviors_modalities_score[behavior][modality] * 100, 2)
                elif modality == 'position-discretized':
                    row['position'] = round(behaviors_modalities_score[behavior][modality] * 100, 2)
                elif modality == 'velocity-discretized':
                    row['velocity'] = round(behaviors_modalities_score[behavior][modality] * 100, 2)
                elif modality == 'torque-discretized':
                    row['torque'] = round(behaviors_modalities_score[behavior][modality] * 100, 2)
                elif modality == 'force-discretized':
                    row['force'] = round(behaviors_modalities_score[behavior][modality] * 100, 2)
                elif modality == 'camera_rgb_image_raw':
                    row['camera_rgb'] = round(behaviors_modalities_score[behavior][modality] * 100, 2)
                else:
                    row[modality] = round(behaviors_modalities_score[behavior][modality] * 100, 2)
            df = df.append(row, ignore_index=True)

    print("df: ", df)
    os.makedirs(results_path, exist_ok=True)
    df.to_csv(results_path + os.sep + DETECTION_TASK + '_' + CLF_NAME + '.csv', index=False)

    with open(results_path + os.sep + DETECTION_TASK + '_' + CLF_NAME + '.csv', 'a') as f:
        writer = csv.writer(f, lineterminator="\n")

        row = ['Average: ']
        for column in df:
            if column != 'behavior':
                row.append(round(df[[column]].mean(axis=0)[column], 2))
        writer.writerow(row)

        writer.writerow(['all_behaviors_modalities: ', round(behaviors_modalities_score['all_behaviors_modalities'] * 100, 2),
                         'all_behaviors_modalities_train: ', round(behaviors_modalities_score['all_behaviors_modalities_train'] * 100, 2),
                         'all_behaviors_modalities_test: ', round(behaviors_modalities_score['all_behaviors_modalities_test'] * 100, 2)])
