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
        # torch._six.string_classes has been removed after pytorch 2.0
        # if not isinstance(path, torch._six.string_classes):
        if not isinstance(path, str):
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
    df1 = pd.read_csv(parent_path+'/datasets/celeba/list_attr_celeba.txt', sep="\s+", skiprows=1, usecols=['Male','Black_Hair','Blond_Hair','Brown_Hair','Gray_Hair'])
    # Make 0 (female) & 1 (male) labels instead of -1 & 1
    df1.loc[df1['Male'] == -1, 'Male'] = 0

    df1.loc[df1['Black_Hair'] == -1, 'Black_Hair'] = 0
    df1.loc[df1['Blond_Hair'] == -1, 'Blond_Hair'] = 0
    df1.loc[df1['Brown_Hair'] == -1, 'Brown_Hair'] = 0
    df1.loc[df1['Gray_Hair'] == -1, 'Gray_Hair'] = 0

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



def old_strategy():
    # strategy1: select the performance with the acc close to the stage1 avg_acc
    acc_dist = abs(stage2_cache[:,0] - avg_stage1_acc)
    closest_idx = np.argpartition(-acc_dist, -win_size)[-win_size:]
    # TNRD -- TNR Difference
    [sg1_avg_Acc, sg1_avg_AD, sg1_avg_EOD, sg1_avg_TNRD, sg1_avg_AOD] = stage2_cache[closest_idx].mean(axis=0).tolist()
    [sg1_std_Acc, sg1_std_AD, sg1_std_EOD, sg1_std_TNRD, sg1_std_AOD] = stage2_cache[closest_idx].std(axis=0).tolist()


    # strategy2: select the best performance with the acc above the the stage1 avg_acc
    above_stage1_acc_idx = np.where(stage2_cache[:,0] - avg_stage1_acc > 0)[0]
    strategy2_FLAG = False
    if len(above_stage1_acc_idx) > 0:
        K = min(win_size, len(above_stage1_acc_idx))
        strategy2_FLAG = True
        above_stage1_acc_stage2_performance = stage2_cache[above_stage1_acc_idx]
        # from large to small
        topK_Acc = np.array(sorted(above_stage1_acc_stage2_performance[:, 0], reverse=True)[:K])
        sg2_avg_Acc, sg2_std_Acc = topK_Acc.mean(), topK_Acc.std()
        topK_AD = np.array(sorted(above_stage1_acc_stage2_performance[:, 1], reverse=False)[:K])
        sg2_avg_AD, sg2_std_AD = topK_AD.mean(), topK_AD.std()
        topK_EOD = np.array(sorted(above_stage1_acc_stage2_performance[:, 2], reverse=False)[:K])
        sg2_avg_EOD, sg2_std_EOD = topK_EOD.mean(), topK_EOD.std()
        top_TNRD = np.array(sorted(above_stage1_acc_stage2_performance[:, 3], reverse=False)[:K])
        sg2_avg_TNRD, sg2_std_TNRD = top_TNRD.mean(), top_TNRD.std()
        topK_AOD = np.array(sorted(above_stage1_acc_stage2_performance[:, 4], reverse=False)[:K])
        sg2_avg_AOD, sg2_std_AOD = topK_AOD.mean(), topK_AOD.std()


    # strategy3: when the AD, EOD, TNRD, AOD below their corresponding stage1 value, the best Acc
    below_stage1_AD_idx = np.where(stage2_cache[:,1] - avg_stage1_AD < 0)[0]
    strategy3_AD_FLAG = False
    if len(below_stage1_AD_idx) > 0:
        strategy3_AD_FLAG = True
        k = min(win_size, len(below_stage1_AD_idx))
        _stage2_cache = stage2_cache[below_stage1_AD_idx, 0]
        topk_idx = np.argpartition(_stage2_cache, -k)[-k:]
        Avg_Acc_with_AD_constraint, Std_Acc_with_AD_constraint = stage2_cache[below_stage1_AD_idx][topk_idx, 0].mean(), stage2_cache[below_stage1_AD_idx][topk_idx, 0].std()
        Avg_AD_with_AD_constraint, Std_AD_with_AD_constraint = stage2_cache[below_stage1_AD_idx][topk_idx, 1].mean(), stage2_cache[below_stage1_AD_idx][topk_idx, 1].std()

    below_stage1_EOD_idx = np.where(stage2_cache[:,2] - avg_stage1_EOD < 0)[0]
    strategy3_EOD_FLAG = False
    if len(below_stage1_EOD_idx) > 0:
        strategy3_EOD_FLAG = True
        k = min(win_size, len(below_stage1_EOD_idx))
        _stage2_cache = stage2_cache[below_stage1_EOD_idx, 0]
        topk_idx = np.argpartition(_stage2_cache, -k)[-k:]
        Avg_Acc_with_EOD_constraint, Std_Acc_with_EOD_constraint = stage2_cache[below_stage1_EOD_idx][topk_idx, 0].mean(), stage2_cache[below_stage1_EOD_idx][topk_idx, 0].std()
        Avg_EOD_with_EOD_constraint, Std_EOD_with_EOD_constraint = stage2_cache[below_stage1_EOD_idx][topk_idx, 2].mean(), stage2_cache[below_stage1_EOD_idx][topk_idx, 2].std()

    below_stage1_TNRD_idx = np.where(stage2_cache[:,3] - avg_stage1_TNRD < 0)[0]
    strategy3_TNRD_FLAG = False
    if len(below_stage1_TNRD_idx) > 0:
        strategy3_TNRD_FLAG = True
        k = min(win_size, len(below_stage1_TNRD_idx))
        _stage2_cache = stage2_cache[below_stage1_TNRD_idx, 0]
        topk_idx = np.argpartition(_stage2_cache, -k)[-k:]
        Avg_Acc_with_TNRD_constraint, Std_Acc_with_TNRD_constraint = stage2_cache[below_stage1_TNRD_idx][topk_idx, 0].mean(), stage2_cache[below_stage1_TNRD_idx][topk_idx, 0].std()
        Avg_TNRD_with_TNRD_constraint, Std_TNRD_with_TNRD_constraint = stage2_cache[below_stage1_TNRD_idx][topk_idx, 3].mean(), stage2_cache[below_stage1_TNRD_idx][topk_idx, 3].std()
        # Acc_with_TNRD_constraint = np.argsort(stage2_cache[below_stage1_TNRD_idx, 0], axis=0)[-max(win_size, len(below_stage1_TNRD_idx)):]

    below_stage1_AOD_idx = np.where(stage2_cache[:,4] - avg_stage1_AOD < 0)[0]
    strategy3_AOD_FLAG = False
    if len(below_stage1_AOD_idx) > 0:
        strategy3_AOD_FLAG = True
        k = min(win_size, len(below_stage1_AOD_idx))
        _stage2_cache = stage2_cache[below_stage1_AOD_idx, 0]
        topk_idx = np.argpartition(_stage2_cache, -k)[-k:]
        Avg_Acc_with_AOD_constraint, Std_Acc_with_AOD_constraint = stage2_cache[below_stage1_AOD_idx][topk_idx, 0].mean(), stage2_cache[below_stage1_AOD_idx][topk_idx, 0].std()
        Avg_AOD_with_AOD_constraint, Std_AOD_with_AOD_constraint = stage2_cache[below_stage1_AOD_idx][topk_idx, 4].mean(), stage2_cache[below_stage1_AOD_idx][topk_idx, 4].std()
        # Acc_with_AOD_constraint = np.argsort(stage2_cache[below_stage1_AOD_idx, 0], axis=0)[-max(win_size, len(below_stage1_AOD_idx)):]

    title = args.title
    time = args.time

    with open(dir_path + '/result.tsv', 'a+') as file:
        file.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(title, time, 'stage1', str(avg_stage1_acc), str(avg_stage1_AD),
                                                             str(avg_stage1_EOD), str(avg_stage1_TNRD), str(avg_stage1_AOD)))

        file.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(title, time, 'sg1', str(sg1_avg_Acc), str(sg1_avg_AD),
                                                             str(sg1_avg_EOD), str(sg1_avg_TNRD), str(sg1_avg_AOD)))
        if strategy2_FLAG:
            file.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(title, time, 'sg2', str(sg2_avg_Acc), str(sg2_avg_AD),
                                                                 str(sg2_avg_EOD), str(sg2_avg_TNRD), str(sg2_avg_AOD)))

        if strategy3_AD_FLAG:
            file.write('{}\t{}\t{}\t{}\t{}\n'.format(title, time, 'sg3-AD', str(Avg_Acc_with_AD_constraint), str(Avg_AD_with_AD_constraint) ))

        if strategy3_EOD_FLAG:
            file.write('{}\t{}\t{}\t{}\t{}\n'.format(title, time, 'sg3-EOD', str(Avg_Acc_with_EOD_constraint), str(Avg_EOD_with_EOD_constraint) ))

        if strategy3_TNRD_FLAG:
            file.write('{}\t{}\t{}\t{}\t{}\n'.format(title, time, 'sg3-TNRD', str(Avg_Acc_with_TNRD_constraint), str(Avg_TNRD_with_TNRD_constraint) ))

        if strategy3_AOD_FLAG:
            file.write('{}\t{}\t{}\t{}\t{}\n'.format(title, time, 'sg3-AOD', str(Avg_Acc_with_AOD_constraint), str(Avg_AOD_with_AOD_constraint) ))



def save_final_result(args, valid_result_cache, test_result_cache, win_size=5):
    assert len(valid_result_cache) == len(test_result_cache)
    # if 'fair' not in args.eps_type:
    #     return
    eps_point = args.epochs_stage1
    test_interval = args.test_interval
    # get stage 1 performance
    stage1_valid_acc_list, stage1_test_acc_list = [], []
    stage1_valid_AD_list, stage1_test_AD_list = [], []
    stage1_valid_EOD_list, stage1_test_EOD_list = [], []
    stage1_valid_AOD_list, stage1_test_AOD_list = [], []
    stage1_valid_TNRD_list, stage1_test_TNRD_list = [], []
    compute_stage1_FLAG = False

    stage2_test_cache = []
    stage2_valid_cache = []

    def mean_std_func(input_list):
        input_array = np.array(input_list)
        input_array_mean = input_array.mean()
        input_array_std = input_array.std()
        return input_array_mean, input_array_std

    for i,j in zip(valid_result_cache, test_result_cache):
        [valid_epoch, valid_acc, valid_AD, valid_EOD, valid_abs_tnr_diff, valid_AOD] = i
        [test_epoch, test_acc, test_AD, test_EOD, test_abs_tnr_diff, test_AOD] = j
        valid_acc = valid_acc.item()
        test_acc = test_acc.item()
        valid_AD = valid_AD.item()
        test_AD = test_AD.item()
        valid_EOD = valid_EOD.item()
        test_EOD = test_EOD.item()
        valid_abs_tnr_diff = valid_abs_tnr_diff.item()
        test_abs_tnr_diff = test_abs_tnr_diff.item()
        valid_AOD = valid_AOD.item()
        test_AOD = test_AOD.item()

        # stage1:
        if valid_epoch < eps_point and valid_epoch >= eps_point - win_size*test_interval:
            stage1_valid_acc_list.append(valid_acc)
            stage1_test_acc_list.append(test_acc)
            stage1_valid_AD_list.append(valid_AD)
            stage1_test_AD_list.append(test_AD)
            stage1_valid_EOD_list.append(valid_EOD)
            stage1_test_EOD_list.append(test_EOD)
            stage1_valid_TNRD_list.append(valid_abs_tnr_diff)
            stage1_test_TNRD_list.append(test_abs_tnr_diff)
            stage1_valid_AOD_list.append(valid_AOD)
            stage1_test_AOD_list.append(test_AOD)

            stage1_cache = {'acc':[stage1_valid_acc_list, stage1_test_acc_list],
                            'AD':[stage1_valid_AD_list, stage1_test_AD_list],
                            'EOD':[stage1_valid_EOD_list, stage1_test_EOD_list],
                            'TNRD':[stage1_valid_TNRD_list, stage1_test_TNRD_list],
                            'AOD':[stage1_valid_AOD_list, stage1_test_AOD_list]}

        if valid_epoch >= eps_point and not compute_stage1_FLAG:
            compute_stage1_FLAG = True
            avg_stage1_valid_acc, std_stage1_valid_acc = mean_std_func(stage1_valid_acc_list)
            avg_stage1_test_acc, std_stage1_test_acc = mean_std_func(stage1_test_acc_list)
            avg_stage1_valid_AD, std_stage1_valid_AD = mean_std_func(stage1_valid_AD_list)
            avg_stage1_test_AD, std_stage1_test_AD = mean_std_func(stage1_test_AD_list)
            avg_stage1_valid_EOD, std_stage1_valid_EOD = mean_std_func(stage1_valid_EOD_list)
            avg_stage1_test_EOD, std_stage1_test_EOD = mean_std_func(stage1_test_EOD_list)
            avg_stage1_valid_TNRD, std_stage1_valid_TNRD = mean_std_func(stage1_valid_TNRD_list)
            avg_stage1_test_TNRD, std_stage1_test_TNRD = mean_std_func(stage1_test_TNRD_list)
            avg_stage1_valid_AOD, std_stage1_valid_AOD = mean_std_func(stage1_valid_AOD_list)
            avg_stage1_test_AOD, std_stage1_test_AOD = mean_std_func(stage1_test_AOD_list)

        if valid_epoch >= eps_point:
            stage2_valid_cache.append((valid_acc, valid_AD, valid_EOD, valid_abs_tnr_diff, valid_AOD))
            stage2_test_cache.append((test_acc, test_AD, test_EOD, test_abs_tnr_diff, test_AOD))

    stage2_test_cache = np.array(stage2_test_cache)
    stage2_valid_cache = np.array(stage2_valid_cache)


    # strategy1: select the performance with the acc close to the stage1 avg_acc
    # TNRD -- TNR Difference
    acc_dist = abs(stage2_valid_cache[:,0] - avg_stage1_valid_acc)
    closest_idx = np.argpartition(-acc_dist, -win_size)[-win_size:]
    min_closest_idx = min(closest_idx)
    [sg1_avg_Acc, sg1_avg_AD, sg1_avg_EOD, sg1_avg_TNRD, sg1_avg_AOD] = stage2_test_cache[min_closest_idx:min_closest_idx+win_size].mean(axis=0).tolist()
    [sg1_std_Acc, sg1_std_AD, sg1_std_EOD, sg1_std_TNRD, sg1_std_AOD] = stage2_test_cache[min_closest_idx:min_closest_idx+win_size].std(axis=0).tolist()


    # strategy2: the best performance with the max acc
    max_acc_idx = np.argpartition(stage2_valid_cache[:,0], -win_size)[-win_size:]
    # [sg2_avg_Acc, sg2_avg_AD, sg2_avg_EOD, sg2_avg_TNRD, sg2_avg_AOD] = stage2_test_cache[max_acc_idx].mean(axis=0).tolist()
    # [sg2_std_Acc, sg2_std_AD, sg2_std_EOD, sg2_std_TNRD, sg2_std_AOD] = stage2_test_cache[max_acc_idx].std(axis=0).tolist()
    min_max_acc_idx = min(max_acc_idx)
    [sg2_avg_Acc, sg2_avg_AD, sg2_avg_EOD, sg2_avg_TNRD, sg2_avg_AOD] = stage2_test_cache[min_max_acc_idx: min_max_acc_idx+win_size].mean(axis=0).tolist()
    [sg2_std_Acc, sg2_std_AD, sg2_std_EOD, sg2_std_TNRD, sg2_std_AOD] = stage2_test_cache[min_max_acc_idx: min_max_acc_idx+win_size].std(axis=0).tolist()


    # strategy3: the performance with stage1_epoch + eps_interval
    end_epoch = min(args.eps_update_interval, args.epochs_stage2)
    [sg3_avg_Acc, sg3_avg_AD, sg3_avg_EOD, sg3_avg_TNRD, sg3_avg_AOD] = stage2_test_cache[end_epoch-win_size:end_epoch].mean(axis=0).tolist()
    [sg3_std_Acc, sg3_std_AD, sg3_std_EOD, sg3_std_TNRD, sg3_std_AOD] = stage2_test_cache[end_epoch-win_size:end_epoch].std(axis=0).tolist()

    # strategy4: the performance at the end of training as stage 1
    end_epoch = min(args.epochs_stage1, args.epochs_stage2)
    [sg4_avg_Acc, sg4_avg_AD, sg4_avg_EOD, sg4_avg_TNRD, sg4_avg_AOD] = stage2_test_cache[end_epoch-win_size:end_epoch].mean(axis=0).tolist()
    [sg4_std_Acc, sg4_std_AD, sg4_std_EOD, sg4_std_TNRD, sg4_std_AOD] = stage2_test_cache[end_epoch-win_size:end_epoch].std(axis=0).tolist()

    # strategy5: the performance at the end of training of all
    end_epoch = args.epochs_stage2 -1
    [sg5_avg_Acc, sg5_avg_AD, sg5_avg_EOD, sg5_avg_TNRD, sg5_avg_AOD] = stage2_test_cache[end_epoch-win_size:end_epoch].mean(axis=0).tolist()
    [sg5_std_Acc, sg5_std_AD, sg5_std_EOD, sg5_std_TNRD, sg5_std_AOD] = stage2_test_cache[end_epoch-win_size:end_epoch].std(axis=0).tolist()


    with open(dir_path + '/new_result_{}.tsv'.format(args.dataset), 'a+') as file:
        file.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(args.title, args.time, 'stage1',
                                     str(avg_stage1_test_acc), str(avg_stage1_test_AD),
                                     str(avg_stage1_test_EOD), str(avg_stage1_test_TNRD), str(avg_stage1_test_AOD)))

        file.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(args.title, args.time, 'sg1',
                                                             str(sg1_avg_Acc), str(sg1_avg_AD),
                                                             str(sg1_avg_EOD), str(sg1_avg_TNRD), str(sg1_avg_AOD)))

        file.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(args.title, args.time, 'sg2',
                                                             str(sg2_avg_Acc), str(sg2_avg_AD),
                                                             str(sg2_avg_EOD), str(sg2_avg_TNRD), str(sg2_avg_AOD)))


        file.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(args.title, args.time, 'sg3',
                                                             str(sg3_avg_Acc), str(sg3_avg_AD),
                                                             str(sg3_avg_EOD), str(sg3_avg_TNRD), str(sg3_avg_AOD)))

        file.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(args.title, args.time, 'sg4',
                                                             str(sg4_avg_Acc), str(sg4_avg_AD),
                                                             str(sg4_avg_EOD), str(sg4_avg_TNRD), str(sg4_avg_AOD)))

        file.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(args.title, args.time, 'sg5',
                                                             str(sg5_avg_Acc), str(sg5_avg_AD),
                                                             str(sg5_avg_EOD), str(sg5_avg_TNRD), str(sg5_avg_AOD)))



