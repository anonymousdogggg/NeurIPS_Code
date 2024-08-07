from __future__ import division

import pickle
import random
import torch
import numpy as np
from torch.nn.modules.loss import _Loss, _WeightedLoss
import torch.nn.functional as F
from torch import Tensor
from typing import Optional

from tqdm import tqdm

import os, sys
import pandas as pd
from collections import defaultdict
from sklearn import feature_extraction, preprocessing
from random import shuffle

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.abspath(os.path.join(dir_path, os.pardir))


def accuracy(output, labels):
    # preds = output.max(1)[1].type_as(labels)
    preds = output
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)



def load_cf10_data(dataset):
    data, targets = [], []

    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]

    if dataset == 'train':
        for file_name, checksum in train_list:
            file_path = os.path.join(parent_path + '/datasets/cifar-10-batches-py', file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')

                data.append(entry['data'])

                if 'labels' in entry:
                    targets.extend(entry['labels'])
                else:
                    targets.extend(entry['fine_labels'])

        data = np.vstack(data).reshape(-1, 3, 32, 32)
        raw_dim = data[0].reshape(-1, ).shape[0]
        data = data.transpose((0, 2, 3, 1))  # convert to HWC
        targets = np.array(targets)
        class_num = max(targets) + 1
    else:
        assert dataset == 'test'
        for file_name, checksum in test_list:
            file_path = os.path.join(parent_path + '/datasets/cifar-10-batches-py', file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')

                data.append(entry['data'])

                if 'labels' in entry:
                    targets.extend(entry['labels'])
                else:
                    targets.extend(entry['fine_labels'])

        data = np.vstack(data).reshape(-1, 3, 32, 32)
        data = data.transpose((0, 2, 3, 1))  # convert to HWC
        targets = np.array(targets)
    return data, targets


def mnist_load_data():
    import codecs

    def get_int(b):
        return int(codecs.encode(b, 'hex'), 16)

    def open_maybe_compressed_file(path):
        """Return a file object that possibly decompresses 'path' on the fly.
           Decompression occurs when argument `path` is a string and ends with '.gz' or '.xz'.
        """
        if not isinstance(path, torch._six.string_classes):
            return path
        if path.endswith('.gz'):
            import gzip
            return gzip.open(path, 'rb')
        if path.endswith('.xz'):
            import lzma
            return lzma.open(path, 'rb')
        return open(path, 'rb')

    def read_sn3_pascalvincent_tensor(path, strict=True):
        """Read a SN3 file in "Pascal Vincent" format (Lush file 'libidx/idx-io.lsh').
           Argument may be a filename, compressed filename, or file object.
        """
        # typemap
        if not hasattr(read_sn3_pascalvincent_tensor, 'typemap'):
            read_sn3_pascalvincent_tensor.typemap = {
                8: (torch.uint8, np.uint8, np.uint8),
                9: (torch.int8, np.int8, np.int8),
                11: (torch.int16, np.dtype('>i2'), 'i2'),
                12: (torch.int32, np.dtype('>i4'), 'i4'),
                13: (torch.float32, np.dtype('>f4'), 'f4'),
                14: (torch.float64, np.dtype('>f8'), 'f8')}
        # read
        with open_maybe_compressed_file(path) as f:
            data = f.read()
        # parse
        magic = get_int(data[0:4])
        nd = magic % 256
        ty = magic // 256
        assert nd >= 1 and nd <= 3
        assert ty >= 8 and ty <= 14
        m = read_sn3_pascalvincent_tensor.typemap[ty]
        s = [get_int(data[4 * (i + 1): 4 * (i + 2)]) for i in range(nd)]
        parsed = np.frombuffer(data, dtype=m[1], offset=(4 * (nd + 1)))
        assert parsed.shape[0] == np.prod(s) or not strict
        return torch.from_numpy(parsed.astype(m[2], copy=False)).view(*s)

    def read_label_file(path):
        with open(path, 'rb') as f:
            x = read_sn3_pascalvincent_tensor(f, strict=False)
        assert (x.dtype == torch.uint8)
        assert (x.ndimension() == 1)
        return x.long()

    def read_image_file(path):
        with open(path, 'rb') as f:
            x = read_sn3_pascalvincent_tensor(f, strict=False)
        assert (x.dtype == torch.uint8)
        assert (x.ndimension() == 3)
        return x

    raw_folder = parent_path+'/datasets/MNIST/raw'
    train=True
    image_file = f"{'train' if train else 't10k'}-images-idx3-ubyte"
    train_data = read_image_file(os.path.join(raw_folder, image_file))

    label_file = f"{'train' if train else 't10k'}-labels-idx1-ubyte"
    train_targets = read_label_file(os.path.join(raw_folder, label_file))

    train=False
    image_file = f"{'train' if train else 't10k'}-images-idx3-ubyte"
    test_data = read_image_file(os.path.join(raw_folder, image_file))

    label_file = f"{'train' if train else 't10k'}-labels-idx1-ubyte"
    test_targets = read_label_file(os.path.join(raw_folder, label_file))


    return (train_data, train_targets), (test_data, test_targets)


def add_intercept(x):
    """ Add intercept to the data before linear classification """
    m,n = x.shape
    intercept = np.ones(m).reshape(m, 1) # the constant b
    return np.concatenate((intercept, x), axis = 1)


def load_german_data():
    data = []
    with open(parent_path+'/datasets/german.data-numeric', 'r') as file:
        for row in file:
            data.append([int(x) for x in row.split()])
    data = np.array(data)
    x = data[:, :-1]
    y = data[:, -1] - 1

    z = []
    with open(parent_path+'/datasets/german.data', 'r') as file:
        for row in file:
            line = [x for x in row.split()]
            if line[8] == 'A92' or line[8] == 'A95':
                z.append(1)
            elif line[8] == 'A91' or line[8] == 'A93' or line[8] == 'A94':
                z.append(0.)
            else:
                print("Wrong gender key!")
                exit(0)
    return x,y, np.array(z)


def load_compas_data():
    # features to be used for classification
    FEATURES_CLASSIFICATION = ["age_cat", "race", "sex", "priors_count", "c_charge_degree"]

    # continuous features, will need to be handled separately from categorical features,
    # categorical features will be encoded using one-hot
    CONT_VARIABLES = ["priors_count"]

    # the decision variable
    CLASS_FEATURE = "two_year_recid"
    SENSITIVE_ATTRS = ["race"]

    COMPAS_INPUT_FILE = parent_path+"/datasets/compas-scores-two-years.csv"

    # load the data and get some stats
    df = pd.read_csv(COMPAS_INPUT_FILE)
    df = df.dropna(subset=["days_b_screening_arrest"])  # dropping missing vals

    # convert to np array
    data = df.to_dict('list')
    for k in data.keys():
        data[k] = np.array(data[k])

    """ Filtering the data """
    # These filters are the same as propublica (refer to https://github.com/propublica/compas-analysis)
    # If the charge date of a defendants Compas scored crime was not within 30 days from when the person was arrested,
    # we assume that because of data quality reasons, that we do not have the right offense.
    idx = np.logical_and(data["days_b_screening_arrest"] <= 30, data["days_b_screening_arrest"] >= -30)

    # We coded the recidivist flag -- is_recid -- to be -1 if we could not find a compas case at all.
    idx = np.logical_and(idx, data["is_recid"] != -1)

    # In a similar vein, ordinary traffic offenses
    # -- those with a c_charge_degree of 'O'
    # -- will not result in Jail time are removed (only two of them).
    idx = np.logical_and(idx, data["c_charge_degree"] != "O")  # F: felony, M: misconduct

    # We filtered the underlying data from Broward county to include only those rows
    # representing people who had either recidivated in two years,
    # or had at least two years outside of a correctional facility.
    idx = np.logical_and(idx, data["score_text"] != "NA")

    # we will only consider blacks and whites for this analysis
    idx = np.logical_and(idx, np.logical_or(data["race"] == "African-American", data["race"] == "Caucasian"))

    # select the examples that satisfy this criteria
    for k in data.keys():
        data[k] = data[k][idx]

    """ Feature normalization and one hot encoding """

    # convert class label 0 to -1
    y = data[CLASS_FEATURE]
    # y[y == 0] = -1

    print("\nNumber of people recidivating within two years")
    print(pd.Series(y).value_counts())
    print()

    # empty array with num rows same as num examples, will hstack the features to it
    X = np.array([]).reshape(len(y), 0)
    x_control = defaultdict(list)

    feature_names = []
    for attr in FEATURES_CLASSIFICATION:
        vals = data[attr]
        if attr in CONT_VARIABLES:
            vals = [float(v) for v in vals]
            # 0 mean and 1 variance
            vals = preprocessing.scale(vals)
            # convert from 1-d arr to a 2-d arr with one col
            vals = np.reshape(vals, (len(y), -1))
        elif attr in SENSITIVE_ATTRS:
            new_val = np.zeros(len(vals))
            for _ in range(len(vals)):
                if vals[_] == 'African-American':
                    new_val[_] = 1.
                elif vals[_] == 'Caucasian':
                    new_val[_] = 0.
                else:
                    print("Wrong race!")
                    exit(0)

            vals = np.reshape(new_val, (len(y), -1))

        else:  # for binary categorical variables, the label binarizer uses just one var instead of two
            lb = preprocessing.LabelBinarizer()
            lb.fit(vals)
            vals = lb.transform(vals)

        # add to sensitive features dict
        if attr in SENSITIVE_ATTRS:
            x_control[attr] = vals

        # add to learnable features
        X = np.hstack((X, vals))

        if attr in CONT_VARIABLES:  # continuous feature, just append the name
            feature_names.append(attr)
        else:  # categorical features
            if vals.shape[1] == 1:  # binary features that passed through lib binarizer
                feature_names.append(attr)
            else:
                for k in lb.classes_:  # non-binary categorical features, need to add the names for each cat
                    feature_names.append(attr + "_" + str(k))

    # convert the sensitive feature to 1-d array
    x_control = dict(x_control)
    for k in x_control.keys():
        assert (x_control[k].shape[1] == 1)  # make sure that the sensitive feature is binary after one hot encoding
        x_control[k] = np.array(x_control[k]).flatten()

    """permute the date randomly"""
    perm = list(range(0, X.shape[0]))
    shuffle(perm)
    X = X[perm]
    y = y[perm]
    for k in x_control.keys():
        x_control[k] = x_control[k][perm]

    X = add_intercept(X)

    feature_names = ["intercept"] + feature_names
    assert (len(feature_names) == X.shape[1])
    print("Features we will be using for classification are:", feature_names, "\n")

    print(X.shape, y.shape, len(x_control['race']))
    return X, y, x_control['race']


# TODO: add args for split ratio
def preprocess_celeba_data(args):
    df1 = pd.read_csv(parent_path+'/datasets/celeba/list_attr_celeba.txt', sep="\s+", skiprows=1, usecols=['Male','Black_Hair','Blond_Hair','Brown_Hair','Gray_Hair', 'Attractive'])
    # Make 0 (female) & 1 (male) labels instead of -1 & 1
    df1.loc[df1['Male'] == -1, 'Male'] = 0

    df1.loc[df1['Black_Hair'] == -1, 'Black_Hair'] = 0
    df1.loc[df1['Blond_Hair'] == -1, 'Blond_Hair'] = 0
    df1.loc[df1['Brown_Hair'] == -1, 'Brown_Hair'] = 0
    df1.loc[df1['Gray_Hair'] == -1, 'Gray_Hair'] = 0
    df1.loc[df1['Attractive'] == -1, 'Attractive'] = 0

    df2 = pd.read_csv(parent_path+'/datasets/celeba/list_eval_partition.txt', sep="\s+", skiprows=0, header=None)
    df2.columns = ['Filename', 'Partition']
    df2 = df2.set_index('Filename')

    df3 = df1.merge(df2, left_index=True, right_index=True)
    df3.to_csv(parent_path+'/datasets/celeba/celeba-gender-partitions.csv')
    df4 = pd.read_csv(parent_path+'/datasets/celeba/celeba-gender-partitions.csv', index_col=0)

    df4.loc[df4['Partition'] == 0].to_csv(parent_path+'/datasets/celeba/celeba-gender-train.csv')
    df4.loc[df4['Partition'] == 1].to_csv(parent_path+'/datasets/celeba/celeba-gender-valid.csv')
    df4.loc[df4['Partition'] == 2].to_csv(parent_path+'/datasets/celeba/celeba-gender-test.csv')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class GumbelAcc(_Loss):
    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(GumbelAcc, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target, type, tau=1, hard=False):
        if type == 'gumbel':
            pred = F.gumbel_softmax(input, tau=tau, hard=hard)
        else:
            pred = F.softmax(input / tau, dim=1)
        acc_loss = pred[torch.nonzero(target)[:,0], torch.nonzero(target)[:,1]] # (pred * target).sum(dim=1)
        acc_loss = acc_loss.sum()
        return acc_loss



class GumbelTPR(_Loss):
    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(GumbelTPR, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target, type, tau=1, hard=False):
        if type == 'gumbel':
            pred = F.gumbel_softmax(input, tau=tau, hard=hard)
        else:
            pred = F.softmax(input / tau, dim=1)

        acc_loss = pred[torch.nonzero(target)[:,0], torch.nonzero(target)[:,1]]
        pos_label_idx = torch.nonzero(target)[:,1] == 1
        # tpr = acc_loss[pos_label_idx]/pos_label_idx.shape[0]
        tpr = acc_loss[pos_label_idx].sum()
        return tpr



class GumbelTNR(_Loss):
    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(GumbelTNR, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target, type, tau=1, hard=False):
        if type == 'gumbel':
            pred = F.gumbel_softmax(input, tau=tau, hard=hard)
        else:
            pred = F.softmax(input / tau, dim=1)

        acc_loss = pred[torch.nonzero(target)[:,0], torch.nonzero(target)[:,1]]
        neg_label_idx = torch.nonzero(target)[:,1] == 0
        # tnr = acc_loss[neg_label_idx]/neg_label_idx.shape[0]
        tnr = acc_loss[neg_label_idx].sum()
        return tnr



class Samplewise_Weighted_CrossEntropyLoss(_WeightedLoss):
    __constants__ = ['ignore_index', 'reduction']
    ignore_index: int

    def __init__(self, weight: Optional[Tensor] = None, size_average=None, ignore_index: int = -100,
                 reduce=None, reduction: str = 'mean') -> None:
        assert reduction in ['mean', 'sum']
        super(Samplewise_Weighted_CrossEntropyLoss, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index

    def forward(self, input: Tensor, target: Tensor, weight: Tensor) -> Tensor:
        non_reduced_loss = F.cross_entropy(input, target, weight=self.weight, ignore_index=self.ignore_index, reduction='none')

        if self.reduction == 'mean':
            return (non_reduced_loss * weight).mean()
        else:
            return (non_reduced_loss * weight).sum()





def save_embedding(args, model, train_dataloader, valid_dataloader, test_dataloader):
    embedding_cache = []
    model.eval()
    print('Saveing embedding...')
    with torch.no_grad():
        for idx, x, _, _ in tqdm(train_dataloader):
            model.zero_grad()
            x = x.to(args.device)
            logit, embedding = model(x)
            embedding_cache.append(embedding.detach().cpu())

        for idx, x, _, _ in tqdm(valid_dataloader):
            model.zero_grad()
            x = x.to(args.device)
            logit, embedding = model(x)
            embedding_cache.append(embedding.detach().cpu())

        for idx, x, _, _ in tqdm(test_dataloader):
            model.zero_grad()
            x = x.to(args.device)
            logit, embedding = model(x)
            embedding_cache.append(embedding.detach().cpu())

    all_embedding = torch.cat(embedding_cache, dim=0)
    with open(parent_path+"/datasets/{}/{}_{}_embedding.pkl".format(args.dataset.lower(), args.dataset, args.label_category), "wb") as output_file:
        pickle.dump(all_embedding, output_file)
    exit(101)


