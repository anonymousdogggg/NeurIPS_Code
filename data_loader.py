import os
import pickle
import torch
from random import shuffle
import pandas as pd
# from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_adult
# from aif360.metrics import BinaryLabelDatasetMetric
from sklearn.preprocessing import MaxAbsScaler, StandardScaler, MinMaxScaler
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import h5py
from skimage import transform
import random
from sklearn import feature_extraction, preprocessing

from myDataset import ColumnarDataset, myMnistuspsDataset, myCifarDataset, CelebaDataset, \
    FairFaceDataset, CImnistDataset
from utils import load_german_data, load_compas_data, load_cf10_data, mnist_load_data, \
    preprocess_celeba_data
import torch.nn.functional as F
from sklearn.model_selection import train_test_split


dir_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.abspath(os.path.join(dir_path, os.pardir))

def cifar_dataloader(args):
    device = args.device
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  # (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)

    train_data, train_targets = load_cf10_data(dataset='train')
    # np.random.shuffle(train_data)
    # np.random.shuffle(train_targets)
    if args.part_data:
        train_data, train_targets = train_data[:20000], train_targets[:20000]

    idx_all = np.array(list(range(train_data.shape[0]))).astype(int)
    shuffle(idx_all)
    split_idx = int(train_data.shape[0] * 0.7)
    idx_train = idx_all[:split_idx]
    idx_valid = idx_all[split_idx:]
    split_train_data, split_train_targets = train_data[idx_train], train_targets[idx_train]
    split_valid_data, split_valid_targets = train_data[idx_valid], train_targets[idx_valid]

    test_data, test_targets = load_cf10_data(dataset='test')

    train_dataset = myCifarDataset((split_train_data, split_train_targets), device, transform=transform,
                                   dataset='train')
    valid_dataset = myCifarDataset((split_valid_data, split_valid_targets), device, transform=transform,
                                   dataset='valid')
    test_dataset = myCifarDataset((test_data, test_targets), device, transform=transform, dataset='test')

    return train_dataset, valid_dataset, test_dataset


def compas_dataloader(args):
    if not os.path.exists('../datasets/compas.pkl'):
        x_data, YL, x_control = load_compas_data()
        if x_control.sum() > 0.5 * len(x_control):
            # reverse the label so that the samples in minority group are labeled as positive
            x_control = (x_control == 0).astype(np.int64)
        group_idx1 = np.where(x_control == 1)[0]
        group_idx0 = np.where(x_control == 0)[0]
        group_1 = list(group_idx1[:int(0.2 * len(group_idx1))])
        if int(len(group_1) * args.ratio) > len(group_idx0):
            length = int(len(group_idx0) / args.ratio)
            group_1 = list(group_idx1[:length])
        label_idx1 = np.where(YL == 1)[0]
        label_idx0 = np.where(YL == 0)[0]
        group_1_normal = len(set(group_1).intersection(label_idx0))
        group_0_idx = list(set(group_idx0).intersection(label_idx0))[:int(group_1_normal * args.ratio)] + list(set(group_idx0).intersection(label_idx1))[:int((len(group_1)-group_1_normal) * args.ratio)]
        idx = group_1 + list(group_idx0[group_0_idx])
        x_data = x_data[idx]
        YL = YL[idx]
        x_control = x_control[idx]

        all_arr = np.arange(x_data.shape[0])
        np.random.shuffle(all_arr)
        x_data = x_data[all_arr]
        YL = YL[all_arr]
        # x_control = x_control[all_arr]

        # idx = list(label_idx1[:int(0.2 * len(label_idx1))]) + list(label_idx0)
        # x_data = x_data[idx]
        # YL = YL[idx]
        # x_control = x_control[idx]

        YL = YL.reshape(-1, )
        # YL = np.concatenate((~YL + 2, YL), 1)
        idx_train = random.sample(range(len(x_data)), int(0.8 * len(x_data)))
        idx_test = list(set(range(len(x_data))) - set(idx_train))
        idx_val = random.sample(idx_train, int(0.2 * len(idx_train)))
        idx_train = list(set(idx_train) - set(idx_val))
        data = {'data': x_data, 'label': YL, 'group': x_control, 'idx_train': idx_train, 'idx_test': idx_test,
                'idx_val': idx_val}
        with open('../datasets/compas_{}.pkl'.format(args.ratio), 'wb') as f:
            pickle.dump(data, f)
    elif args.ratio > 0:
        with open('../datasets/compas_{}.pkl'.format(args.ratio), 'rb') as f:
            data = pickle.load(f)
            x_data, YL, x_control = data['data'], data['label'], data['group']
            idx_train, idx_test, idx_val = data['idx_train'], data['idx_test'], data['idx_val']
    else:
        with open('../datasets/compas.pkl', 'rb') as f:
            data = pickle.load(f)
            x_data, YL, x_control = data['data'], data['label'], data['group']
            idx_train, idx_test, idx_val = data['idx_train'], data['idx_test'], data['idx_val']

    ## normlaize input data into (0, 1)
    if args.normalize == 1:
        scaler = MinMaxScaler()
        scaler.fit(x_data)
        x_data = scaler.transform(x_data)
    all_arr = np.arange(x_data.shape[0])
    np.random.shuffle(all_arr)
    trn_ds = ColumnarDataset(x_data[idx_train, :], YL[idx_train], x_control[idx_train].astype(int), args.device)
    val_ds = ColumnarDataset(x_data[idx_val, :], YL[idx_val], x_control[idx_val].astype(int), args.device)
    test_ds = ColumnarDataset(x_data[idx_test, :], YL[idx_test], x_control[idx_test].astype(int), args.device)
    all_ds = ColumnarDataset(x_data[all_arr], YL[all_arr], x_control[all_arr].astype(int), args.device)
    # train_dataset = myMnistuspsDataset(args, (
    # split_train_data[train_arr], split_train_targets[train_arr], train_group_label[train_arr]), device, transform=transform_func, dataset='train')
    # valid_dataset = myMnistuspsDataset(args, (
    # split_valid_data[valid_arr], split_valid_targets[valid_arr], valid_group_label[valid_arr]), device, transform=transform_func, dataset='valid')
    # test_dataset = myMnistuspsDataset(args, (test_data[test_arr], test_targets[test_arr], test_group_label[test_arr]), device, transform=transform_func, dataset='test')
    # all_dataset = myMnistuspsDataset(args, (all_data[all_arr], all_targets[all_arr], all_group_label[all_arr]), device, transform=transform_func, dataset='all')

    return trn_ds, val_ds, test_ds, all_ds


#
# def celebatab_dataloader(args):
#     device = args.device
#     df = pd.read_csv(parent_path + '/datasets/celeba_baldvsnonbald_normalised.csv')
#     target = df['class'].values
#     x_df = df.drop(['class'], axis=1)
#     z = df['A19'].values
#     group_idx1 = np.where(z == 1)[0]
#     group_idx0 = np.where(z == 0)[0]
#     idx = list(group_idx1[:int(0.1 * len(group_idx1))]) + list(group_idx0)
#     # z = np.where(z == 0, 1, 0)
#     x = x_df.values
#     x = x[idx]
#     z = z[idx]
#     target = target[idx]
#     print("Data shape: (%d, %d)" % x.shape)
#     all_arr = np.arange(x.shape[0])
#     np.random.shuffle(all_arr)
#
#
#     trn_ds = ColumnarDataset(x, target, z, device)
#     val_ds = ColumnarDataset(x, target, z, device)
#     test_ds = ColumnarDataset(x, target, z, device)
#     all_ds = ColumnarDataset(x[all_arr], target[all_arr], z[all_arr], device)
#
#     return trn_ds, val_ds, test_ds, all_ds


def celebaimage_dataloader(args):
    device = args.device
    with open('../datasets/celebA_Blond_Hair_embedding.pkl', 'rb') as f:
        data = pickle.load(f)
    target = []
    groups = []
    # idx = []
    count = 0
    with open('../datasets/list_attr_celeba.txt', 'r') as f:
        _ = f.readline()
        label = f.readline().split()
        lines = f.readlines()
        label_index = label.index('Blond_Hair')
        group_index = label.index('Male')
        for line in lines:
            a = list(map(int, line.split()[1:]))
            target.append(a[label_index])
            groups.append(a[group_index])
            count += 1
            if count >= 50000:
                break
    target = np.array(target)
    target = (target + 1) / 2
    index_0 = np.where(target == 0)[0]
    index_1 = np.where(target == 1)[0]
    index = list(index_1[:int(len(index_1)*0.1)]) + list(index_0)
    groups = np.array(groups)
    groups = (groups + 1) / 2
    data = data[index, :]
    target = target[index]
    groups = groups[index]
    print("Data shape: (%d, %d)" % data.shape)
    all_arr = np.arange(data.shape[0])
    np.random.shuffle(all_arr)
    trn_ds = ColumnarDataset(data, target, groups, device)
    val_ds = ColumnarDataset(data, target, groups, device)
    test_ds = ColumnarDataset(data, target, groups, device)
    all_ds = ColumnarDataset(data[all_arr], target[all_arr], groups[all_arr], device)
    return trn_ds, val_ds, test_ds, all_ds



def celebatab_dataloader(args):
    device = args.device
    if not os.path.exists('../datasets/celebatab.pkl'):
        df = pd.read_csv('../datasets/celebatab.csv')
        target = df['Attractive'].values
        x_df = df.drop(['Attractive'], axis=1)
        z = df['Male'].values
        group_idx1 = np.where(z == 1)[0]
        group_idx0 = np.where(z == 0)[0]
        # idx = list(group_idx1[:int(0.1 * len(group_idx1))]) + list(group_idx0[:int(0.3 * len(group_idx0))])
        idx = list(group_idx1[:int(0.05 * len(group_idx1))]) + list(group_idx0)
        x = x_df.values
        x = x[idx]
        z = z[idx]
        target = target[idx]

        all_arr = np.arange(x.shape[0])
        np.random.shuffle(all_arr)
        x = x[all_arr]
        target = target[all_arr]
        z = z[all_arr]
        # a = np.where(z == 0)[0]
        b = np.where(z == 1)[0]

        label_idx1 = np.where(target == 1)[0]
        label_idx0 = np.where(target == 0)[0]
        idx = list(set(label_idx1[:int(0.05 * len(label_idx1))]).union(b)) + list(label_idx0)
        x = x[idx]
        target = target[idx]
        z = z[idx]



        # all_arr = np.arange(x.shape[0])
        # np.random.shuffle(all_arr)
        # x = x[all_arr]
        # target = target[all_arr]
        # z = z[all_arr]

        data = {'data': x, 'label': target, 'group': z}
        with open('../datasets/celebatab.pkl', 'wb') as f:
            pickle.dump(data, f)
    else:
        with open('../datasets/celebatab.pkl', 'rb') as f:
            data = pickle.load(f)
            x, target, z = data['data'], data['label'], data['group']
            # idx_train, idx_test, idx_val = data['idx_train'], data['idx_test'], data['idx_val']

    print("Data shape: (%d, %d)" % x.shape)
    all_arr = np.arange(x.shape[0])
    np.random.shuffle(all_arr)
    trn_ds = ColumnarDataset(x, target, z, device)
    val_ds = ColumnarDataset(x, target, z, device)
    test_ds = ColumnarDataset(x, target, z, device)
    all_ds = ColumnarDataset(x[all_arr], target[all_arr], z[all_arr], device)

    return trn_ds, val_ds, test_ds, all_ds


def thyroid_dataloader(args):
    device = args.device
    df = pd.read_csv(parent_path + '/datasets/annthyroid_21feat_normalised.csv')
    target = df['class'].values
    x_df = df.drop(['class'], axis=1)
    z = df['Dim_2=0'].values
    z = np.where(z == 0, 1, 0)
    # x_df = x_df.drop(['Dim_2=0'], axis=1)
    x = x_df.values
    print("Data shape: (%d, %d)" % x.shape)

    trn_ds = ColumnarDataset(x, target, z, device)
    val_ds = ColumnarDataset(x, target, z, device)
    test_ds = ColumnarDataset(x, target, z, device)
    all_ds = ColumnarDataset(x, target, z, device)

    return trn_ds, val_ds, test_ds, all_ds


def german_dataloader(args):
    x_data, YL, x_control = load_german_data()
    scaler = preprocessing.MinMaxScaler()  # Default behavior is to scale to [0,1]
    x_data = scaler.fit_transform(x_data)
    YL = YL.reshape(-1, 1)
    YL = np.concatenate((~YL + 2, YL), 1)
    idx_train = random.sample(range(len(x_data)), int(0.8 * len(x_data)))
    idx_test = list(set(range(len(x_data))) - set(idx_train))
    idx_val = random.sample(idx_train, int(0.2 * len(idx_train)))
    idx_train = list(set(idx_train) - set(idx_val))
    trn_ds =  ColumnarDataset(x_data[idx_train, :], YL[idx_train, :], x_control[idx_train].astype(int), args.device)
    val_ds =  ColumnarDataset(x_data[idx_val, :],   YL[idx_val, :],   x_control[idx_val].astype(int),   args.device)
    test_ds = ColumnarDataset(x_data[idx_test, :],  YL[idx_test, :],  x_control[idx_test].astype(int),  args.device)
    return trn_ds, val_ds, test_ds


def folktables_dataloader(args):
    device = args.device
    ACSIncome = folktables.BasicProblem(
        features=[
            'AGEP',
            'COW',
            'SCHL',
            'MAR',
            'OCCP',
            'POBP',
            'RELP',
            'WKHP',
            # 'SEX',
            'RAC1P',
        ],
        target='PINCP',
        target_transform=lambda x: x > 25000,
        group='SEX',
        preprocess=adult_filter,
        postprocess=lambda x: np.nan_to_num(x, -1),
    )

    data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
    # acs_data = data_source.get_data(states=["CA"], download=True)
    acs_data = data_source.get_data(states=["TX"], download=True)
    features, label, group = ACSIncome.df_to_numpy(acs_data)
    label = label.astype('int')
    gp1_idx = np.where(group == 1)
    gp2_idx = np.where(group == 2)
    group[gp1_idx] = 0
    group[gp2_idx] = 1
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    X_train, X_test, y_train, y_test, group_train, group_test = train_test_split(features, label, group, test_size=args.test_ratio, random_state=0)
    X_train, X_valid, y_train, y_valid, group_train, group_valid = train_test_split(X_train, y_train, group_train, test_size=args.valid_ratio, random_state=0)

    train_one_hot = np.zeros((y_train.shape[0], int(y_train.max()) + 1))
    valid_one_hot = np.zeros((y_valid.shape[0], int(y_valid.max()) + 1))
    test_one_hot = np.zeros((y_test.shape[0], int(y_test.max()) + 1))

    train_one_hot[np.arange(y_train.shape[0]), y_train] = 1
    valid_one_hot[np.arange(y_valid.shape[0]), y_valid] = 1
    test_one_hot[np.arange(y_test.shape[0]), y_test] = 1

    trn_ds = ColumnarDataset(X_train, train_one_hot, group_train, device)
    val_ds = ColumnarDataset(X_valid, valid_one_hot, group_valid, device)
    test_ds = ColumnarDataset(X_test, test_one_hot, group_test, device)

    return trn_ds, val_ds, test_ds



def mnistinvert_dataloader(args):
    device = args.device
    transform_func = transforms.Compose([
        transforms.ToTensor()
    ])  # transforms.Normalize((0.1307,), (0.3081,)) for mnist
    (mnist_train_data, mnist_train_targets), (mnist_test_data, mnist_test_targets) = mnist_load_data()

    usps_train_data = 255 - mnist_train_data
    usps_test_data = 255 - mnist_test_data
    usps_train_targets = mnist_train_targets
    usps_test_targets = mnist_test_targets

    mnist_train_data = mnist_train_data / 255.
    mnist_test_data = mnist_test_data / 255.
    usps_train_data = usps_train_data / 255.
    usps_test_data = usps_test_data / 255.

    # zero padding
    mnist_train_data = np.pad(mnist_train_data, ((0, 0), (2, 2), (2, 2)))
    mnist_test_data = np.pad(mnist_test_data, ((0, 0), (2, 2), (2, 2)))
    usps_train_data = np.pad(usps_train_data, ((0, 0), (2, 2), (2, 2)))
    usps_test_data = np.pad(usps_test_data, ((0, 0), (2, 2), (2, 2)))

    class_bins_train = {}
    for i in range(len(mnist_train_data)):
        if int(mnist_train_targets[i]) not in class_bins_train:
            class_bins_train[int(mnist_train_targets[i])] = []
        class_bins_train[int(mnist_train_targets[i])].append(mnist_train_data[i])

    train_split = []
    val_split = []
    val_ratio = 0.1

    for _class, data in class_bins_train.items():
        count = len(data)
        print("(Train set) Class %d has %d samples" % (_class, count))

        if _class == 0:
            # Create a validation set by taking a small portion of data from the training set
            count_per_class = int(count * val_ratio)
            val_split += data[:count_per_class]
            train_split += data[count_per_class::]

    # Create a test split
    class_bins_test = {}
    for i in range(len(mnist_test_data)):
        if int(mnist_test_targets[i]) not in class_bins_test:
            class_bins_test[int(mnist_test_targets[i])] = []
        class_bins_test[int(mnist_test_targets[i])].append(mnist_test_data[i])

    for _class, data in class_bins_test.items():
        count = len(data)
        print("(Test set) Class %d has %d samples" % (_class, count))

    for cls in range(1):  # Use class 0 as the normal class
        print('Creating a test split with inlier class %d' % cls)
        cls_balanced_data_mnist = []
        cls_balanced_target_mnist = []
        cls_cnt = [0] * 10

        # First add all the inlier samples

        for _class, data in class_bins_test.items():
            if _class == cls:
                cls_balanced_data_mnist.extend(data)
                cls_balanced_target_mnist.extend([_class] * len(data))
                cls_cnt[cls] += len(data)

        # Add the same number of outlier samples by sampling from all other classes
        num_inlier = len(cls_balanced_data_mnist)
        ### ------------------------------------------- ###
        # num_outlier = int(num_inlier * args.outlier_ratio)
        num_outlier = int(0.5 * num_inlier * args.outlier_ratio)
        ### ------------------------------------------- ###
        outlier_cnt = 0
        for i in range(len(mnist_test_targets)):
            if mnist_test_targets[i] != cls and outlier_cnt < num_outlier:
                cls_balanced_data_mnist.append(mnist_test_data[i])
                cls_balanced_target_mnist.append(cls+1)
                outlier_cnt += 1
                cls_cnt[mnist_test_targets[i]] += 1

        print('Number of samples for each class:  ', cls_cnt)

    class_bins_usps_train = {}
    ### ------------------------------------------- ###
    # for i in range(len(usps_train_data)):
    for i in range(int(0.05 * len(usps_train_data))):
    ### ------------------------------------------- ###
        if int(usps_train_targets[i]) not in class_bins_usps_train:
            class_bins_usps_train[int(usps_train_targets[i])] = []
        class_bins_usps_train[int(usps_train_targets[i])].append(usps_train_data[i])

    train_usps_split = []
    val_usps_split = []
    val_ratio = 0.1

    for _class, data in class_bins_usps_train.items():
        count = len(data)
        print("(Train set) Class %d has %d samples" % (_class, count))

        # Create a validation set by taking a small portion of data from the training set
        if _class == 0:
            count_per_class = int(count * val_ratio)
            val_usps_split += data[:count_per_class]
            train_usps_split += data[count_per_class::]

    # Create a test split
    class_bins_usps_test = {}
    ### ------------------------------------------- ###
    # for i in range(len(usps_test_data)):
    for i in range(int(0.1 * len(usps_test_data))):
    ### ------------------------------------------- ###

        if int(usps_test_targets[i]) not in class_bins_usps_test:
            class_bins_usps_test[int(usps_test_targets[i])] = []
        class_bins_usps_test[int(usps_test_targets[i])].append(usps_test_data[i])

    for _class, data in class_bins_usps_test.items():
        count = len(data)
        print("(Test set) Class %d has %d samples" % (_class, count))

    for cls in range(1):  # Use class 0 as the normal class
        print('Creating a test split with inlier class %d' % cls)
        cls_balanced_data_usps = []
        cls_balanced_target_usps = []
        cls_cnt = [0] * 10

        # First add all the inlier samples

        for _class, data in class_bins_usps_test.items():
            if _class == cls:
                cls_balanced_data_usps.extend(data)
                cls_balanced_target_usps.extend([_class] * len(data))
                cls_cnt[cls] += len(data)

        # Add the same number of outlier samples by sampling from all other classes
        num_inlier = len(cls_balanced_data_usps)
        ### ------------------------------------------- ###
        num_outlier = int(0.5 * num_inlier * args.outlier_ratio)
        ### ------------------------------------------- ###
        outlier_cnt = 0
        for i in range(len(usps_test_targets)):
            if usps_test_targets[i] != cls and outlier_cnt < num_outlier:
                cls_balanced_data_usps.append(usps_test_data[i])
                cls_balanced_target_usps.append(cls+1)
                outlier_cnt += 1
                cls_cnt[usps_test_targets[i]] += 1

        print('Number of samples for each class:  ', cls_cnt)

    # 3.create group label for train, valid, test.
    split_train_data = np.concatenate((train_split, train_usps_split))
    split_train_targets = np.zeros(len(split_train_data))
    train_group_label = np.zeros(split_train_data.shape[0])
    train_group_label[len(train_split):] = 1

    split_valid_data = np.concatenate((val_split, val_usps_split))
    split_valid_targets = np.zeros(len(split_valid_data))
    valid_group_label = np.zeros(split_valid_data.shape[0])
    valid_group_label[len(val_split):] = 1

    test_data = np.concatenate((cls_balanced_data_mnist, cls_balanced_data_usps))
    test_targets = np.concatenate((cls_balanced_target_mnist, cls_balanced_target_usps))
    test_group_label = np.zeros(test_data.shape[0])
    test_group_label[len(cls_balanced_target_mnist):] = 1

    all_data = np.concatenate((split_train_data, split_valid_data, test_data))
    all_targets = np.concatenate((split_train_targets, split_valid_targets, test_targets))
    all_group_label = np.concatenate((train_group_label, valid_group_label, test_group_label))

    train_arr = np.arange(split_train_data.shape[0])
    valid_arr = np.arange(split_valid_data.shape[0])
    test_arr = np.arange(test_data.shape[0])
    all_arr = np.arange(all_data.shape[0])
    np.random.shuffle(train_arr)
    np.random.shuffle(valid_arr)
    np.random.shuffle(test_arr)
    np.random.shuffle(all_arr)

    train_dataset = myMnistuspsDataset(args, (
        split_train_data[train_arr], split_train_targets[train_arr], train_group_label[train_arr]), device,
                                       transform=transform_func, dataset='train')
    valid_dataset = myMnistuspsDataset(args, (
        split_valid_data[valid_arr], split_valid_targets[valid_arr], valid_group_label[valid_arr]), device,
                                       transform=transform_func, dataset='valid')
    test_dataset = myMnistuspsDataset(args, (test_data[test_arr], test_targets[test_arr], test_group_label[test_arr]),
                                      device, transform=transform_func, dataset='test')
    all_dataset = myMnistuspsDataset(args, (all_data[all_arr], all_targets[all_arr], all_group_label[all_arr]), device,
                                     transform=transform_func, dataset='all')

    return train_dataset, valid_dataset, test_dataset, all_dataset

    # train_arr = np.arange(split_train_data.shape[0])
    # valid_arr = np.arange(split_valid_data.shape[0])
    # test_arr = np.arange(test_data.shape[0])
    # np.random.shuffle(train_arr)
    # np.random.shuffle(valid_arr)
    # np.random.shuffle(test_arr)
    #
    # train_dataset = myMnistuspsDataset(args, (
    #     split_train_data[train_arr], split_train_targets[train_arr], train_group_label[train_arr]), device,
    #                                    transform=transform_func, dataset='train')
    # valid_dataset = myMnistuspsDataset(args, (
    #     split_valid_data[valid_arr], split_valid_targets[valid_arr], valid_group_label[valid_arr]), device,
    #                                    transform=transform_func, dataset='valid')
    # test_dataset = myMnistuspsDataset(args, (test_data[test_arr], test_targets[test_arr], test_group_label[test_arr]),
    #                                   device, transform=transform_func, dataset='test')
    #
    # return train_dataset, valid_dataset, test_dataset

#
# def mnistplususps_dataloader(args):
#     device = args.device
#     transform_func = transforms.Compose([
#         transforms.ToTensor()
#     ])  # transforms.Normalize((0.1307,), (0.3081,)) for mnist
#     (mnist_train_data, mnist_train_targets), (mnist_test_data, mnist_test_targets) = mnist_load_data()
#     mnist_train_data = mnist_train_data / 255.
#     mnist_test_data = mnist_test_data / 255.
#     # zero padding
#     mnist_train_data = np.pad(mnist_train_data, ((0, 0), (2, 2), (2, 2)))
#     mnist_test_data = np.pad(mnist_test_data, ((0, 0), (2, 2), (2, 2)))
#
#     class_bins_train = {}
#     for i in range(len(mnist_train_data)):
#         if int(mnist_train_targets[i]) not in class_bins_train:
#             class_bins_train[int(mnist_train_targets[i])] = []
#         class_bins_train[int(mnist_train_targets[i])].append(mnist_train_data[i])
#
#     train_split = []
#     val_split = []
#     val_ratio = 0.1
#
#     for _class, data in class_bins_train.items():
#         count = len(data)
#         print("(Train set) Class %d has %d samples" % (_class, count))
#
#         if _class == 0:
#         # Create a validation set by taking a small portion of data from the training set
#             count_per_class = int(count * val_ratio)
#             val_split += data[:count_per_class]
#             train_split += data[count_per_class::]
#
#     # Create a test split
#     class_bins_test = {}
#     for i in range(int(0.5 * len(mnist_test_data))):
#         if int(mnist_test_targets[i]) not in class_bins_test:
#             class_bins_test[int(mnist_test_targets[i])] = []
#         class_bins_test[int(mnist_test_targets[i])].append(mnist_test_data[i])
#
#     for _class, data in class_bins_test.items():
#         count = len(data)
#         print("(Test set) Class %d has %d samples" % (_class, count))
#
#
#     for cls in range(1):  # Use class 0 as the normal class
#         print('Creating a test split with inlier class %d' % cls)
#         cls_balanced_data_mnist = []
#         cls_balanced_target_mnist = []
#         cls_cnt = [0] * 10
#
#         # First add all the inlier samples
#
#         for _class, data in class_bins_test.items():
#             if _class == cls:
#                 cls_balanced_data_mnist.extend(data)
#                 cls_balanced_target_mnist.extend([_class] * len(data))
#                 cls_cnt[cls] += len(data)
#
#         # Add the same number of outlier samples by sampling from all other classes
#         num_inlier = len(cls_balanced_data_mnist)
#         num_outlier = int(num_inlier * args.outlier_ratio)
#         outlier_cnt = 0
#         for i in range(len(mnist_test_targets)):
#             if mnist_test_targets[i] != cls and outlier_cnt < num_outlier:
#                 cls_balanced_data_mnist.append(mnist_test_data[i])
#                 cls_balanced_target_mnist.append(cls+1)
#                 outlier_cnt += 1
#                 cls_cnt[mnist_test_targets[i]] += 1
#
#
#         print('Number of samples for each class:  ', cls_cnt)
#
#
#     # 2.load usps
#     with h5py.File(os.path.join(parent_path + '/datasets/usps', 'usps.h5'), 'r') as hf:
#         train = hf.get('train')
#         usps_train_data = train.get('data')[:]
#         usps_train_targets = train.get('target')[:]
#         test = hf.get('test')
#         usps_test_data = test.get('data')[:]
#         usps_test_targets = test.get('target')[:]
#
#     usps_train_data = usps_train_data.reshape(-1, 16, 16)
#     usps_test_data = usps_test_data.reshape(-1, 16, 16)
#     usps_train_data = np.array(
#         [transform.resize(usps_train_data[i].reshape(16, 16), (32, 32)) for i in range(len(usps_train_data))])
#     usps_test_data = np.array(
#         [transform.resize(usps_test_data[i].reshape(16, 16), (32, 32)) for i in range(len(usps_test_data))])
#
#     class_bins_usps_train = {}
#     # for i in range(len(usps_train_data)):
#     ### ------------------------------------------- ###
#     # for i in range(len(usps_test_data)):
#     for i in range(int(0.2 * len(usps_train_data))):
#     ### ------------------------------------------- ###
#         if int(usps_train_targets[i]) not in class_bins_usps_train:
#             class_bins_usps_train[int(usps_train_targets[i])] = []
#         class_bins_usps_train[int(usps_train_targets[i])].append(usps_train_data[i])
#
#     train_usps_split = []
#     val_usps_split = []
#     val_ratio = 0.1
#
#     for _class, data in class_bins_usps_train.items():
#         count = len(data)
#         print("(Train set) Class %d has %d samples" % (_class, count))
#
#         # Create a validation set by taking a small portion of data from the training set
#         if _class == 0:
#             count_per_class = int(count * val_ratio)
#             val_usps_split += data[:count_per_class]
#             train_usps_split += data[count_per_class::]
#
#     # Create a test split
#     class_bins_usps_test = {}
#     ### ------------------------------------------- ###
#     # for i in range(len(usps_test_data)):
#     for i in range(int(0.1 * len(usps_test_data))):
#     ### ------------------------------------------- ###
#         if int(usps_test_targets[i]) not in class_bins_usps_test:
#             class_bins_usps_test[int(usps_test_targets[i])] = []
#         class_bins_usps_test[int(usps_test_targets[i])].append(usps_test_data[i])
#
#     for _class, data in class_bins_usps_test.items():
#         count = len(data)
#         print("(Test set) Class %d has %d samples" % (_class, count))
#
#     for cls in range(1):  # Use class 0 as the normal class, the abnormal samples are marked as 1
#         print('Creating a test split with inlier class %d' % cls)
#         cls_balanced_data_usps = []
#         cls_balanced_target_usps = []
#         cls_cnt = [0] * 10
#
#         # First add all the inlier samples
#
#         for _class, data in class_bins_usps_test.items():
#             if _class == cls:
#                 cls_balanced_data_usps.extend(data)
#                 cls_balanced_target_usps.extend([_class] * len(data))
#                 cls_cnt[cls] += len(data)
#
#         # Add the same number of outlier samples by sampling from all other classes
#         num_inlier = len(cls_balanced_data_usps)
#         num_outlier = int(num_inlier * args.outlier_ratio)
#         outlier_cnt = 0
#         for i in range(len(usps_test_targets)):
#             if usps_test_targets[i] != cls and outlier_cnt < num_outlier:
#                 cls_balanced_data_usps.append(usps_test_data[i])
#                 cls_balanced_target_usps.append(cls+1)
#                 outlier_cnt += 1
#                 cls_cnt[usps_test_targets[i]] += 1
#
#
#         print('Number of samples for each class:  ', cls_cnt)
#
#     # 3.create group label for train, valid, test.
#     split_train_data = np.concatenate((train_split, train_usps_split))
#     split_train_targets = np.zeros(len(split_train_data))
#     train_group_label = np.zeros(split_train_data.shape[0])
#     train_group_label[len(train_split):] = 1
#
#     split_valid_data = np.concatenate((val_split, val_usps_split))
#     split_valid_targets = np.zeros(len(split_valid_data))
#     valid_group_label = np.zeros(split_valid_data.shape[0])
#     valid_group_label[len(val_split):] = 1
#
#     test_data = np.concatenate((cls_balanced_data_mnist, cls_balanced_data_usps))
#     test_targets = np.concatenate((cls_balanced_target_mnist, cls_balanced_target_usps))
#     test_group_label = np.zeros(test_data.shape[0])
#     test_group_label[len(cls_balanced_target_mnist):] = 1
#
#     all_data = np.concatenate((split_train_data, split_valid_data, test_data))
#     all_targets = np.concatenate((split_train_targets, split_valid_targets, test_targets))
#     all_group_label = np.concatenate((train_group_label, valid_group_label, test_group_label))
#
#     train_arr = np.arange(split_train_data.shape[0])
#     valid_arr = np.arange(split_valid_data.shape[0])
#     test_arr = np.arange(test_data.shape[0])
#     all_arr = np.arange(all_data.shape[0])
#     np.random.shuffle(train_arr)
#     np.random.shuffle(valid_arr)
#     np.random.shuffle(test_arr)
#     np.random.shuffle(all_arr)
#
#     train_dataset = myMnistuspsDataset(args, (
#     split_train_data[train_arr], split_train_targets[train_arr], train_group_label[train_arr]), device, transform=transform_func, dataset='train')
#     valid_dataset = myMnistuspsDataset(args, (
#     split_valid_data[valid_arr], split_valid_targets[valid_arr], valid_group_label[valid_arr]), device, transform=transform_func, dataset='valid')
#     test_dataset = myMnistuspsDataset(args, (test_data[test_arr], test_targets[test_arr], test_group_label[test_arr]), device, transform=transform_func, dataset='test')
#     all_dataset = myMnistuspsDataset(args, (all_data[all_arr], all_targets[all_arr], all_group_label[all_arr]), device, transform=transform_func, dataset='all')
#
#     return train_dataset, valid_dataset, test_dataset, all_dataset
#


def mnistplususps_dataloader(args):
    if args.ratio > 0:
        with open('../datasets/mnistandusps_{}.pkl'.format(args.ratio), 'rb') as f:
            data = pickle.load(f)
        x_data = data['data']
        YL = data['label']
        x_control = data['group']

        device = args.device
        transform_func = transforms.Compose([
            transforms.ToTensor()
        ])  # transforms.Normalize((0.1307,), (0.3081,)) for mnist
        all_arr = np.arange(x_data.shape[0])
        np.random.shuffle(all_arr)
        all_dataset = myMnistuspsDataset(args, (x_data[all_arr], YL[all_arr], x_control[all_arr]), device,
                                         transform=transform_func, dataset='all')
        return all_dataset, all_dataset, all_dataset, all_dataset
    else:
        device = args.device
        transform_func = transforms.Compose([
            transforms.ToTensor()
        ])  # transforms.Normalize((0.1307,), (0.3081,)) for mnist
        (mnist_train_data, mnist_train_targets), (mnist_test_data, mnist_test_targets) = mnist_load_data()
        mnist_train_data = mnist_train_data / 255.
        mnist_test_data = mnist_test_data / 255.
        # zero padding
        mnist_train_data = np.pad(mnist_train_data, ((0, 0), (2, 2), (2, 2)))
        mnist_test_data = np.pad(mnist_test_data, ((0, 0), (2, 2), (2, 2)))

        class_bins_train = {}
        for i in range(len(mnist_train_data)):
            if int(mnist_train_targets[i]) not in class_bins_train:
                class_bins_train[int(mnist_train_targets[i])] = []
            class_bins_train[int(mnist_train_targets[i])].append(mnist_train_data[i])

        train_split = []
        val_split = []
        val_ratio = 0.1

        for _class, data in class_bins_train.items():
            count = len(data)
            print("(Train set) Class %d has %d samples" % (_class, count))

            if _class == 0:
            # Create a validation set by taking a small portion of data from the training set
                count_per_class = int(count * val_ratio)
                val_split += data[:count_per_class]
                train_split += data[count_per_class::]

        # Create a test split
        class_bins_test = {}
        for i in range(len(mnist_test_data)):
            if int(mnist_test_targets[i]) not in class_bins_test:
                class_bins_test[int(mnist_test_targets[i])] = []
            class_bins_test[int(mnist_test_targets[i])].append(mnist_test_data[i])

        for _class, data in class_bins_test.items():
            count = len(data)
            print("(Test set) Class %d has %d samples" % (_class, count))


        for cls in range(1):  # Use class 0 as the normal class
            print('Creating a test split with inlier class %d' % cls)
            cls_balanced_data_mnist = []
            cls_balanced_target_mnist = []
            cls_cnt = [0] * 10

            # First add all the inlier samples

            for _class, data in class_bins_test.items():
                if _class == cls:
                    cls_balanced_data_mnist.extend(data)
                    cls_balanced_target_mnist.extend([_class] * len(data))
                    cls_cnt[cls] += len(data)

            # Add the same number of outlier samples by sampling from all other classes
            num_inlier = len(cls_balanced_data_mnist)
            num_outlier = int(num_inlier * args.outlier_ratio)
            outlier_cnt = 0
            for i in range(len(mnist_test_targets)):
                if mnist_test_targets[i] != cls and outlier_cnt < num_outlier:
                    cls_balanced_data_mnist.append(mnist_test_data[i])
                    cls_balanced_target_mnist.append(cls+1)
                    outlier_cnt += 1
                    cls_cnt[mnist_test_targets[i]] += 1


            print('Number of samples for each class:  ', cls_cnt)


        # 2.load usps
        with h5py.File(os.path.join(parent_path + '/datasets/usps', 'usps.h5'), 'r') as hf:
            train = hf.get('train')
            usps_train_data = train.get('data')[:]
            usps_train_targets = train.get('target')[:]
            test = hf.get('test')
            usps_test_data = test.get('data')[:]
            usps_test_targets = test.get('target')[:]

        usps_train_data = usps_train_data.reshape(-1, 16, 16)
        usps_test_data = usps_test_data.reshape(-1, 16, 16)
        usps_train_data = np.array(
            [transform.resize(usps_train_data[i].reshape(16, 16), (32, 32)) for i in range(len(usps_train_data))])
        usps_test_data = np.array(
            [transform.resize(usps_test_data[i].reshape(16, 16), (32, 32)) for i in range(len(usps_test_data))])

        class_bins_usps_train = {}
        ### ------------------------------------------- ###
        for i in range(len(usps_train_data)):
        # for i in range(int(0.2 * len(usps_train_data))):
        ### ------------------------------------------- ###
            if int(usps_train_targets[i]) not in class_bins_usps_train:
                class_bins_usps_train[int(usps_train_targets[i])] = []
            class_bins_usps_train[int(usps_train_targets[i])].append(usps_train_data[i])

        train_usps_split = []
        val_usps_split = []
        val_ratio = 0.1

        for _class, data in class_bins_usps_train.items():
            count = len(data)
            print("(Train set) Class %d has %d samples" % (_class, count))

            # Create a validation set by taking a small portion of data from the training set
            if _class == 0:
                count_per_class = int(count * val_ratio)
                val_usps_split += data[:count_per_class]
                train_usps_split += data[count_per_class::]

        # Create a test split
        class_bins_usps_test = {}
        ### ------------------------------------------- ###
        for i in range(len(usps_test_data)):
        # for i in range(int(0.1 * len(usps_test_data))):
        ### ------------------------------------------- ###
            if int(usps_test_targets[i]) not in class_bins_usps_test:
                class_bins_usps_test[int(usps_test_targets[i])] = []
            class_bins_usps_test[int(usps_test_targets[i])].append(usps_test_data[i])

        for _class, data in class_bins_usps_test.items():
            count = len(data)
            print("(Test set) Class %d has %d samples" % (_class, count))

        for cls in range(1):  # Use class 0 as the normal class, the abnormal samples are marked as 1
            print('Creating a test split with inlier class %d' % cls)
            cls_balanced_data_usps = []
            cls_balanced_target_usps = []
            cls_cnt = [0] * 10

            # First add all the inlier samples

            for _class, data in class_bins_usps_test.items():
                if _class == cls:
                    cls_balanced_data_usps.extend(data)
                    cls_balanced_target_usps.extend([_class] * len(data))
                    cls_cnt[cls] += len(data)

            # Add the same number of outlier samples by sampling from all other classes
            num_inlier = len(cls_balanced_data_usps)
            num_outlier = int(num_inlier * args.outlier_ratio)
            outlier_cnt = 0
            for i in range(len(usps_test_targets)):
                if usps_test_targets[i] != cls and outlier_cnt < num_outlier:
                    cls_balanced_data_usps.append(usps_test_data[i])
                    cls_balanced_target_usps.append(cls+1)
                    outlier_cnt += 1
                    cls_cnt[usps_test_targets[i]] += 1


            print('Number of samples for each class:  ', cls_cnt)

        # 3.create group label for train, valid, test.
        split_train_data = np.concatenate((train_split, train_usps_split))
        split_train_targets = np.zeros(len(split_train_data))
        train_group_label = np.zeros(split_train_data.shape[0])
        train_group_label[len(train_split):] = 1

        split_valid_data = np.concatenate((val_split, val_usps_split))
        split_valid_targets = np.zeros(len(split_valid_data))
        valid_group_label = np.zeros(split_valid_data.shape[0])
        valid_group_label[len(val_split):] = 1

        test_data = np.concatenate((cls_balanced_data_mnist, cls_balanced_data_usps))
        test_targets = np.concatenate((cls_balanced_target_mnist, cls_balanced_target_usps))
        test_group_label = np.zeros(test_data.shape[0])
        test_group_label[len(cls_balanced_target_mnist):] = 1

        all_data = np.concatenate((split_train_data, split_valid_data, test_data))
        all_targets = np.concatenate((split_train_targets, split_valid_targets, test_targets))
        all_group_label = np.concatenate((train_group_label, valid_group_label, test_group_label))

        train_arr = np.arange(split_train_data.shape[0])
        valid_arr = np.arange(split_valid_data.shape[0])
        test_arr = np.arange(test_data.shape[0])
        all_arr = np.arange(all_data.shape[0])
        np.random.shuffle(train_arr)
        np.random.shuffle(valid_arr)
        np.random.shuffle(test_arr)
        np.random.shuffle(all_arr)

        train_dataset = myMnistuspsDataset(args, (
        split_train_data[train_arr], split_train_targets[train_arr], train_group_label[train_arr]), device, transform=transform_func, dataset='train')
        valid_dataset = myMnistuspsDataset(args, (
        split_valid_data[valid_arr], split_valid_targets[valid_arr], valid_group_label[valid_arr]), device, transform=transform_func, dataset='valid')
        test_dataset = myMnistuspsDataset(args, (test_data[test_arr], test_targets[test_arr], test_group_label[test_arr]), device, transform=transform_func, dataset='test')
        all_dataset = myMnistuspsDataset(args, (all_data[all_arr], all_targets[all_arr], all_group_label[all_arr]), device, transform=transform_func, dataset='all')

        return train_dataset, valid_dataset, test_dataset, all_dataset



def celeba_dataloader(args):
    device = args.device
    # Note that transforms.ToTensor()
    # already divides pixels by 255. internally
    custom_transform = transforms.Compose([transforms.CenterCrop((178, 178)),
                                           transforms.Resize((128, 128)),
                                           # transforms.Grayscale(),
                                           # transforms.Lambda(lambda x: x/255.),
                                           transforms.ToTensor()])

    train_csv_path = parent_path + '/datasets/celeba/celeba-gender-train.csv'
    valid_csv_path = parent_path + '/datasets/celeba/celeba-gender-valid.csv'
    test_csv_path = parent_path + '/datasets/celeba/celeba-gender-test.csv'

    # orig: 0.1 for test and valid, 0.8 for train
    train_df = pd.read_csv(train_csv_path, index_col=0)
    valid_df = pd.read_csv(valid_csv_path, index_col=0)
    test_df = pd.read_csv(test_csv_path, index_col=0)

    all = len(valid_df) + len(train_df) + len(test_df)

    if args.resplit:
        assert args.valid_ratio < 0.1 # 0.05 0.025 0.01
        valid_df, valid_to_train_df = np.split(valid_df.sample(frac=1, random_state=42), [  int(args.valid_ratio/(len(valid_df)/all) * len(valid_df))     ])
        train_df = pd.concat([train_df, valid_to_train_df])
        print(len(train_df)/all, len(valid_df)/all, len(test_df)/all)
        # train_ratio = 1 - args.valid_ratio - args.test_ratio
        # df = pd.concat([train_df, valid_df, test_df])
        # train_df, valid_df, test_df = np.split(df.sample(frac=1, random_state=42), [int(train_ratio * len(df)), int((train_ratio+args.valid_ratio) * len(df))])

    train_dataset = CelebaDataset(args, train_df, device=device, transform=custom_transform, label_category=args.label_category)
    valid_dataset = CelebaDataset(args, valid_df, device=device, transform=custom_transform, label_category=args.label_category)
    test_dataset = CelebaDataset(args, test_df, device=device, transform=custom_transform, label_category=args.label_category)

    train_label0_idx_set = set(torch.where(train_dataset.label_y==0)[0].to('cpu').tolist())
    train_label1_idx_set = set(torch.where(train_dataset.label_y==1)[0].to('cpu').tolist())
    train_group1_idx_set = set(train_dataset.group1_idx)
    train_group2_idx_set = set(train_dataset.group2_idx)

    valid_label0_idx_set = set(torch.where(valid_dataset.label_y==0)[0].to('cpu').tolist())
    valid_label1_idx_set = set(torch.where(valid_dataset.label_y==1)[0].to('cpu').tolist())
    valid_group1_idx_set = set(valid_dataset.group1_idx)
    valid_group2_idx_set = set(valid_dataset.group2_idx)

    test_label0_idx_set = set(torch.where(test_dataset.label_y==0)[0].to('cpu').tolist())
    test_label1_idx_set = set(torch.where(test_dataset.label_y==1)[0].to('cpu').tolist())
    test_group1_idx_set = set(test_dataset.group1_idx)
    test_group2_idx_set = set(test_dataset.group2_idx)

    train_group1_label0_idx_set = train_group1_idx_set.intersection(train_label0_idx_set)
    train_group1_label1_idx_set = train_group1_idx_set.intersection(train_label1_idx_set)
    train_group2_label0_idx_set = train_group2_idx_set.intersection(train_label0_idx_set)
    train_group2_label1_idx_set = train_group2_idx_set.intersection(train_label1_idx_set)

    valid_group1_label0_idx_set = valid_group1_idx_set.intersection(valid_label0_idx_set)
    valid_group1_label1_idx_set = valid_group1_idx_set.intersection(valid_label1_idx_set)
    valid_group2_label0_idx_set = valid_group2_idx_set.intersection(valid_label0_idx_set)
    valid_group2_label1_idx_set = valid_group2_idx_set.intersection(valid_label1_idx_set)

    test_group1_label0_idx_set = test_group1_idx_set.intersection(test_label0_idx_set)
    test_group1_label1_idx_set = test_group1_idx_set.intersection(test_label1_idx_set)
    test_group2_label0_idx_set = test_group2_idx_set.intersection(test_label0_idx_set)
    test_group2_label1_idx_set = test_group2_idx_set.intersection(test_label1_idx_set)

    print(len(train_group1_label0_idx_set) + len(valid_group1_label0_idx_set) + len(test_group1_label0_idx_set))
    print(len(train_group1_label1_idx_set) + len(valid_group1_label1_idx_set) + len(test_group1_label1_idx_set))
    print(len(train_group2_label0_idx_set) + len(valid_group2_label0_idx_set) + len(test_group2_label0_idx_set))
    print(len(train_group2_label1_idx_set) + len(valid_group2_label1_idx_set) + len(test_group2_label1_idx_set))

    return train_dataset, valid_dataset, test_dataset


def fairface_dataloader(args):
    device = args.device
    from sklearn.preprocessing import LabelEncoder
    import pandas as pd
    data_folder = parent_path+'/datasets/fairface'
    train_label_path = parent_path+'/datasets/fairface/fairface_label_train.csv'
    val_label_path = parent_path+'/datasets/fairface/fairface_label_val.csv'
    train_df = pd.read_csv(train_label_path)
    val_df = pd.read_csv(val_label_path)
    df = pd.concat([train_df, val_df])
    age_category = sorted(list(set(df['age'].values.tolist())))
    gender_category = sorted(list(set(df['gender'].values.tolist())))
    race_category = sorted(list(set(df['race'].values.tolist())))
    category_list = [age_category, gender_category, race_category]

    attribute_mapping = {}
    for idx, attribute in enumerate(['age', 'gender', 'race']):
        attri_category = category_list[idx]
        le = LabelEncoder()
        le.fit(attri_category)
        attribute_mapping[attribute]=le.classes_
        df[attribute] = le.transform(df[attribute].values)


    if args.resplit:
        train_ratio = 1 - args.valid_ratio - args.test_ratio
        train, validate, test = np.split(df.sample(frac=1, random_state=args.seed), [int(train_ratio * len(df)), int((train_ratio+args.valid_ratio) * len(df))])
    else:
        # 20% for valid, 20% for test
        train, validate, test = np.split(df.sample(frac=1, random_state=42), [int(.6 * len(df)), int(.8 * len(df))])

    if args.save_eps:
        with open(parent_path+'/saved_model/train_set/fairface.pkl', 'wb') as file:
            pickle.dump((train, attribute_mapping), file)

    train_dataset = FairFaceDataset(args, root=data_folder, label_category=args.label_category,
                                    df=train, train=True, device=device, attribute_mapping=attribute_mapping)
    valid_dataset = FairFaceDataset(args, root=data_folder, label_category=args.label_category,
                                    df=validate, train=False, device=device, attribute_mapping=attribute_mapping)
    test_dataset = FairFaceDataset(args, root=data_folder, label_category=args.label_category,
                                   df=test, train=False, device=device, attribute_mapping=attribute_mapping)

    train_label0_idx_set = set(torch.where(train_dataset.label_y == 0)[0].to('cpu').tolist())
    train_label1_idx_set = set(torch.where(train_dataset.label_y == 1)[0].to('cpu').tolist())
    train_group1_idx_set = set(train_dataset.group1_idx)
    train_group2_idx_set = set(train_dataset.group2_idx)

    valid_label0_idx_set = set(torch.where(valid_dataset.label_y == 0)[0].to('cpu').tolist())
    valid_label1_idx_set = set(torch.where(valid_dataset.label_y == 1)[0].to('cpu').tolist())
    valid_group1_idx_set = set(valid_dataset.group1_idx)
    valid_group2_idx_set = set(valid_dataset.group2_idx)

    test_label0_idx_set = set(torch.where(test_dataset.label_y == 0)[0].to('cpu').tolist())
    test_label1_idx_set = set(torch.where(test_dataset.label_y == 1)[0].to('cpu').tolist())
    test_group1_idx_set = set(test_dataset.group1_idx)
    test_group2_idx_set = set(test_dataset.group2_idx)

    train_group1_label0_idx_set = train_group1_idx_set.intersection(train_label0_idx_set)
    train_group1_label1_idx_set = train_group1_idx_set.intersection(train_label1_idx_set)
    train_group2_label0_idx_set = train_group2_idx_set.intersection(train_label0_idx_set)
    train_group2_label1_idx_set = train_group2_idx_set.intersection(train_label1_idx_set)

    valid_group1_label0_idx_set = valid_group1_idx_set.intersection(valid_label0_idx_set)
    valid_group1_label1_idx_set = valid_group1_idx_set.intersection(valid_label1_idx_set)
    valid_group2_label0_idx_set = valid_group2_idx_set.intersection(valid_label0_idx_set)
    valid_group2_label1_idx_set = valid_group2_idx_set.intersection(valid_label1_idx_set)

    test_group1_label0_idx_set = test_group1_idx_set.intersection(test_label0_idx_set)
    test_group1_label1_idx_set = test_group1_idx_set.intersection(test_label1_idx_set)
    test_group2_label0_idx_set = test_group2_idx_set.intersection(test_label0_idx_set)
    test_group2_label1_idx_set = test_group2_idx_set.intersection(test_label1_idx_set)

    print(len(train_group1_label0_idx_set) + len(valid_group1_label0_idx_set) + len(test_group1_label0_idx_set))
    print(len(train_group1_label1_idx_set) + len(valid_group1_label1_idx_set) + len(test_group1_label1_idx_set))
    print(len(train_group2_label0_idx_set) + len(valid_group2_label0_idx_set) + len(test_group2_label0_idx_set))
    print(len(train_group2_label1_idx_set) + len(valid_group2_label1_idx_set) + len(test_group2_label1_idx_set))


    return train_dataset, valid_dataset, test_dataset



def dataloader_wrapper(trn_ds, val_ds, test_ds, all_ds=None, train_batch_size=1, valid_batch_size=1, test_batch_size=1,
                       all_batch_size=1):
    train_dataloader, valid_dataloader, test_dataloader, all_dataloader = None, None, None, None
    if trn_ds is not None:
        train_dataloader = DataLoader(trn_ds, train_batch_size, shuffle=True)
    if val_ds is not None:
        valid_dataloader = DataLoader(val_ds, valid_batch_size, shuffle=False)
    if test_ds is not None:
        test_dataloader = DataLoader(test_ds, test_batch_size, shuffle=False)
    if all_ds is not None:
        all_dataloader = DataLoader(all_ds, all_batch_size, shuffle=False)
    return trn_ds, val_ds, test_ds, train_dataloader, valid_dataloader, test_dataloader, all_dataloader


def load_data(args):
    all_ds = None
    if args.dataset == 'compas':
        trn_ds, val_ds, test_ds, all_ds = compas_dataloader(args)
        args.raw_dim = trn_ds.data.shape[1]
        print((len(trn_ds), len(val_ds), len(test_ds), len(all_ds), args.raw_dim)) # (3694, 1584, 1584, 8)
        args.k = 1000
        args.weight_decay = 0.0001
    elif args.dataset == 'celebaimage':
        trn_ds, val_ds, test_ds, all_ds = celebaimage_dataloader(args)
        example_data = trn_ds.data[0]
        args.raw_dim = example_data.reshape(-1).shape[0]
        args.k = 4000
    elif args.dataset == 'mnistandusps':
        trn_ds, val_ds, test_ds, all_ds = mnistplususps_dataloader(args)
        example_data = trn_ds.data[0]
        args.raw_dim = example_data.reshape(-1).shape[0]
        args.k = 1000
        args.weight_decay = 0.0001
    elif args.dataset == 'mnistinvert':
        trn_ds, val_ds, test_ds, all_ds = mnistinvert_dataloader(args)
        example_data = trn_ds.data[0]
        args.raw_dim = example_data.reshape(-1).shape[0]
        args.k = 400
        args.weight_decay = 0.001
    elif args.dataset == 'thyroid':
        trn_ds, val_ds, test_ds, all_ds = thyroid_dataloader(args)
        example_data = trn_ds.data[0]
        args.raw_dim = example_data.reshape(-1).shape[0]
        args.k = 400

    elif args.dataset == 'celebatab':
        trn_ds, val_ds, test_ds, all_ds = celebatab_dataloader(args)
        example_data = trn_ds.data[0]
        args.raw_dim = example_data.reshape(-1).shape[0]
        args.k = 4000
        args.weight_decay = 0.0001
    elif args.dataset == 'mnistandusps_bin':
        trn_ds, val_ds, test_ds = mnistplususps_dataloader(args)
        example_data = trn_ds.data[0]
        args.raw_dim = example_data.reshape(-1).shape[0]
        args.num_class = 2
        trn_ds.label_y = trn_ds.label_y % 2
        val_ds.label_y = val_ds.label_y % 2
        test_ds.label_y = test_ds.label_y % 2
        trn_ds.y = F.one_hot(trn_ds.label_y, num_classes=args.num_class)
        val_ds.y = F.one_hot(val_ds.label_y, num_classes=args.num_class)
        test_ds.y = F.one_hot(test_ds.label_y, num_classes=args.num_class)
    elif args.dataset == 'celebA':
        preprocess_celeba_data(args)
        trn_ds, val_ds, test_ds, all_ds = celeba_dataloader(args)
        args.raw_dim = 178 * 218
        args.k = 10000

    elif args.dataset == 'fairface':
        trn_ds, val_ds, test_ds = fairface_dataloader(args)
        args.num_class = trn_ds.y.shape[1]
    elif args.dataset == 'clr_mnist':
        trn_ds, val_ds, test_ds = CI_minst_dataloader(args)
        args.raw_dim = trn_ds.data[0].reshape(1,-1).shape[1]
        args.num_class = 2
    else:
        trn_ds, val_ds, test_ds = cifar_dataloader(args)

    # post process
    if args.dataset in ['celebA', 'fairface'] and args.model=='mlp':
        label_cate_name = args.label_category if  args.dataset=='celebA' else 'age'
        with open(parent_path+"/datasets/{}/{}_{}_embedding.pkl".format(args.dataset.lower(), args.dataset, label_cate_name), 'rb') as file:
            pretrain_embedding = pickle.load(file)
        assert pretrain_embedding.shape[0] == len(trn_ds)+len(val_ds)+len(test_ds)

        if args.dataset == 'celebA':
            trn_ds.pretrain_data = pretrain_embedding[:len(trn_ds)]
            val_ds.pretrain_data = pretrain_embedding[len(trn_ds):len(trn_ds)+len(val_ds)]
            test_ds.pretrain_data = pretrain_embedding[len(trn_ds)+len(val_ds):]
        elif args.dataset == 'fairface':
            trn_ds.pretrain_data = pretrain_embedding[list(trn_ds.df.index)]
            val_ds.pretrain_data = pretrain_embedding[list(val_ds.df.index)]
            test_ds.pretrain_data = pretrain_embedding[list(test_ds.df.index)]

        args.raw_dim = pretrain_embedding.shape[1]
        # trn_ds.remove_last()
        # val_ds.remove_last()
        # test_ds.remove_last()

    if not hasattr(trn_ds, 'group_indicator'):
        num_train = trn_ds.y.shape[0]
        group_indicator = -1 * np.ones(num_train)
        group_indicator[trn_ds.group1_idx] = 0
        group_indicator[trn_ds.group2_idx] = 1
        trn_ds.group_indicator = group_indicator

    if not hasattr(val_ds, 'group_indicator'):
        num_valid = val_ds.y.shape[0]
        group_indicator = -1 * np.ones(num_valid)
        group_indicator[val_ds.group1_idx] = 0
        group_indicator[val_ds.group2_idx] = 1
        val_ds.group_indicator = group_indicator

    if not hasattr(test_ds, 'group_indicator'):
        num_test = test_ds.y.shape[0]
        group_indicator = -1 * np.ones(num_test)
        group_indicator[test_ds.group1_idx] = 0
        group_indicator[test_ds.group2_idx] = 1
        test_ds.group_indicator = group_indicator

    if not hasattr(all_ds, 'group_indicator'):
        num_test = test_ds.y.shape[0]
        group_indicator = -1 * np.ones(num_test)
        group_indicator[test_ds.group1_idx] = 0
        group_indicator[test_ds.group2_idx] = 1
        test_ds.group_indicator = group_indicator

    # if args.optim == 'GD':
    #     num_train = len(trn_ds)
    #     num_val = len(val_ds)
    #     num_test = len(test_ds)
    #     num_all = len(all_ds)
    #     trn_ds, val_ds, test_ds, train_dataloader, valid_dataloader, test_dataloader, all_dataloader \
    #         = dataloader_wrapper(trn_ds, val_ds, test_ds, all_ds, num_train, num_val, num_test, num_all)
    # else:
    #     trn_ds, val_ds, test_ds, train_dataloader, valid_dataloader, test_dataloader, all_dataloader \
    #         = dataloader_wrapper(trn_ds, val_ds, test_ds, all_ds, args.train_batch_size,
    #                             args.valid_batch_size, args.test_batch_size, args.train_batch_size)

    # return trn_ds, val_ds, test_ds, train_dataloader, valid_dataloader, test_dataloader, all_dataloader
    return trn_ds, val_ds, test_ds, all_ds


# def load_data(args):
#     all_ds = None
#     if args.dataset == 'adults':
#         trn_ds, val_ds, test_ds = load_aif360_data(args)
#         args.raw_dim = trn_ds.features.shape[1]
#         print((len(trn_ds), len(val_ds), len(test_ds), args.raw_dim)) # (20021, 4400, 24421, 18)
#         args.num_class = 2
#     elif args.dataset =='folktable':
#         trn_ds, val_ds, test_ds = folktables_dataloader(args)
#         args.raw_dim = trn_ds.features.shape[1]
#         print((len(trn_ds), len(val_ds), len(test_ds), args.raw_dim)) # (125225, 31307, 39133, 10)
#         args.num_class = 2
#     elif args.dataset == 'german':
#         trn_ds, val_ds, test_ds = german_dataloader(args)
#         args.raw_dim = trn_ds.features.shape[1]
#         print((len(trn_ds), len(val_ds), len(test_ds), args.raw_dim)) # (700, 300, 300, 24)
#         args.num_class = 2
#     elif args.dataset == 'compas':
#         trn_ds, val_ds, test_ds, all_ds = compas_dataloader(args)
#         args.raw_dim = trn_ds.features.shape[1]
#         print((len(trn_ds), len(val_ds), len(test_ds), args.raw_dim)) # (3694, 1584, 1584, 8)
#         args.num_class = 2
#         args.k = 350
#     elif args.dataset == 'mnistandusps':
#         trn_ds, val_ds, test_ds, all_ds = mnistplususps_dataloader(args)
#         example_data = trn_ds.data[0]
#         args.raw_dim = example_data.reshape(-1).shape[0]
#         args.k = 1000
#     elif args.dataset == 'mnistinvert':
#         trn_ds, val_ds, test_ds, all_ds = mnistinvert_dataloader(args)
#         example_data = trn_ds.data[0]
#         args.raw_dim = example_data.reshape(-1).shape[0]
#         args.k = 400
#
#     elif args.dataset == 'mnistandusps_bin':
#         trn_ds, val_ds, test_ds = mnistplususps_dataloader(args)
#         example_data = trn_ds.data[0]
#         args.raw_dim = example_data.reshape(-1).shape[0]
#         args.num_class = 2
#         trn_ds.label_y = trn_ds.label_y % 2
#         val_ds.label_y = val_ds.label_y % 2
#         test_ds.label_y = test_ds.label_y % 2
#         trn_ds.y = F.one_hot(trn_ds.label_y, num_classes=args.num_class)
#         val_ds.y = F.one_hot(val_ds.label_y, num_classes=args.num_class)
#         test_ds.y = F.one_hot(test_ds.label_y, num_classes=args.num_class)
#     elif args.dataset == 'celebA':
#         preprocess_celeba_data(args)
#         trn_ds, val_ds, test_ds = celeba_dataloader(args)
#         args.num_class = trn_ds.y.shape[1]
#     elif args.dataset == 'fairface':
#         trn_ds, val_ds, test_ds = fairface_dataloader(args)
#         args.num_class = trn_ds.y.shape[1]
#     elif args.dataset == 'clr_mnist':
#         trn_ds, val_ds, test_ds = CI_minst_dataloader(args)
#         args.raw_dim = trn_ds.data[0].reshape(1,-1).shape[1]
#         args.num_class = 2
#     else:
#         trn_ds, val_ds, test_ds = cifar_dataloader(args)
#
#     # post process
#     if args.dataset in ['celebA', 'fairface'] and args.model=='mlp':
#         label_cate_name = args.label_category if  args.dataset=='celebA' else 'age'
#         with open(parent_path+"/datasets/{}/{}_{}_embedding.pkl".format(args.dataset.lower(), args.dataset, label_cate_name), 'rb') as file:
#             pretrain_embedding = pickle.load(file)
#         assert pretrain_embedding.shape[0] == len(trn_ds)+len(val_ds)+len(test_ds)
#
#         if args.dataset == 'celebA':
#             trn_ds.pretrain_data = pretrain_embedding[:len(trn_ds)]
#             val_ds.pretrain_data = pretrain_embedding[len(trn_ds):len(trn_ds)+len(val_ds)]
#             test_ds.pretrain_data = pretrain_embedding[len(trn_ds)+len(val_ds):]
#         elif args.dataset == 'fairface':
#             trn_ds.pretrain_data = pretrain_embedding[list(trn_ds.df.index)]
#             val_ds.pretrain_data = pretrain_embedding[list(val_ds.df.index)]
#             test_ds.pretrain_data = pretrain_embedding[list(test_ds.df.index)]
#
#         args.raw_dim = pretrain_embedding.shape[1]
#         # trn_ds.remove_last()
#         # val_ds.remove_last()
#         # test_ds.remove_last()
#
#     if not hasattr(trn_ds, 'group_indicator'):
#         num_train = trn_ds.y.shape[0]
#         group_indicator = -1 * np.ones(num_train)
#         group_indicator[trn_ds.group1_idx] = 0
#         group_indicator[trn_ds.group2_idx] = 1
#         trn_ds.group_indicator = group_indicator
#
#     if not hasattr(val_ds, 'group_indicator'):
#         num_valid = val_ds.y.shape[0]
#         group_indicator = -1 * np.ones(num_valid)
#         group_indicator[val_ds.group1_idx] = 0
#         group_indicator[val_ds.group2_idx] = 1
#         val_ds.group_indicator = group_indicator
#
#     if not hasattr(test_ds, 'group_indicator'):
#         num_test = test_ds.y.shape[0]
#         group_indicator = -1 * np.ones(num_test)
#         group_indicator[test_ds.group1_idx] = 0
#         group_indicator[test_ds.group2_idx] = 1
#         test_ds.group_indicator = group_indicator
#
#     if not hasattr(all_ds, 'group_indicator'):
#         num_test = test_ds.y.shape[0]
#         group_indicator = -1 * np.ones(num_test)
#         group_indicator[test_ds.group1_idx] = 0
#         group_indicator[test_ds.group2_idx] = 1
#         test_ds.group_indicator = group_indicator
#
#     if args.optim == 'GD':
#         num_train = len(trn_ds)
#         num_val = len(val_ds)
#         num_test = len(test_ds)
#         num_all = len(all_ds)
#         trn_ds, val_ds, test_ds, train_dataloader, valid_dataloader, test_dataloader, all_dataloader \
#             = dataloader_wrapper(trn_ds, val_ds, test_ds, all_ds, num_train, num_val, num_test, num_all)
#     else:
#         trn_ds, val_ds, test_ds, train_dataloader, valid_dataloader, test_dataloader, all_dataloader \
#             = dataloader_wrapper(trn_ds, val_ds, test_ds, all_ds, args.train_batch_size,
#                                 args.valid_batch_size, args.test_batch_size, args.train_batch_size)
#
#     return trn_ds, val_ds, test_ds, train_dataloader, valid_dataloader, test_dataloader, all_dataloader

