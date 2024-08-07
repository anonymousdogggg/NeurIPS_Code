from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.datasets.vision import VisionDataset
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import pandas as pd
import os


dir_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.abspath(os.path.join(dir_path, os.pardir))

class ColumnarDataset(Dataset):
    def __init__(self, features, y, sex_feat, device):
        num_data = sex_feat.shape[0]
        self.group_label = sex_feat
        self.group2_idx_set = set(list(sex_feat.nonzero()[0]))
        self.group1_idx_set = set(list(range(num_data))) - self.group2_idx_set
        self.group1_idx = np.array(list(self.group1_idx_set)).astype(int)
        self.group2_idx = np.array(list(self.group2_idx_set)).astype(int)

        self.data = torch.FloatTensor(features).to(device)
        self.y = torch.tensor(y, dtype=torch.long).to(device)
        # self.label_y = torch.where(self.y)[1]
        self.label_y = self.y

        self.idxes = torch.tensor(list(range(num_data)))

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return self.idxes[idx], self.data[idx], self.y[idx], self.label_y[idx], self.group_label[idx]


class myCifarDataset(VisionDataset):

    def __init__(self, data, device, root=None, transform=None, target_transform=None, dataset=None):
        assert root is not None or dataset is not None

        super(myCifarDataset, self).__init__(root, transform=transform,
                                             target_transform=target_transform)
        self.transform = transform
        self.target_transform = target_transform

        (self.data, self.targets) = data
        self.y = np.zeros((self.targets.shape[0], int(self.targets.max()) + 1))
        self.y[np.arange(self.targets.shape[0]), self.targets.astype(int)] = 1
        self.y = torch.tensor(self.y, dtype=torch.long).to(device)

        self.idxs = torch.tensor(range(self.data.shape[0]))
        self.group1_idx = np.where(self.targets == 3)[0]  # cat
        self.group2_idx = np.where(self.targets == 5)[0]  # dog
        self.group3_idx = np.where(self.targets == 0)[0]  # airplane
        self.group4_idx = np.where(self.targets == 1)[0]  # automobile
        self.group5_idx = np.where(self.targets == 2)[0]  # bird
        self.group6_idx = np.where(self.targets == 4)[0]  # deer
        self.group7_idx = np.where(self.targets == 6)[0]  # frog
        self.group8_idx = np.where(self.targets == 7)[0]  # horse
        self.group9_idx = np.where(self.targets == 8)[0]  # ship
        self.group0_idx = np.where(self.targets == 9)[0]  # truck

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target, idx_array = self.data[index], self.y[index], self.idxs[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return idx_array, img, target


class myMnistuspsDataset(VisionDataset):

    def __init__(self, args, data, device, root=None, transform=None, target_transform=None, dataset=None):
        assert root is not None or dataset is not None

        super(myMnistuspsDataset, self).__init__(root, transform=transform,
                                                 target_transform=target_transform)
        self.args = args
        self.transform = transform
        self.target_transform = target_transform
        (self.data, self.targets, self.group_label) = data
        self.y = np.zeros((self.targets.shape[0], int(self.targets.max()) + 1))
        self.y[np.arange(self.targets.shape[0]), self.targets.astype(int)] = 1
        self.y = torch.tensor(self.y, dtype=torch.long).to(device)
        self.label_y = torch.where(self.y)[1]
        self.idxs = torch.tensor(range(self.data.shape[0]))
        self.group1_idx = np.where(self.group_label==0)[0]
        self.group2_idx = np.where(self.group_label==1)[0]
        # self.group1_idx = np.arange(pos)
        # self.group2_idx = np.arange(pos, len(self.data))
        self.device = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target, idx_array = self.data[index], self.y[index], self.idxs[index]
        batch_label_y = self.label_y[index]
        group_label = torch.tensor(self.group_label[index], dtype=torch.long).to(self.device)
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return idx_array, img, target, batch_label_y, group_label

    def remove_last(self):
        num_data = self.y.shape[0]
        need_remove_num = num_data % self.args.tracker_bz
        if need_remove_num == 0:
            return

        self.y = self.y[0:-need_remove_num]
        self.label_y = self.label_y[0:-need_remove_num]
        self.pretrain_data = self.pretrain_data[0:-need_remove_num]
        self.idxs = torch.tensor(range(self.y.shape[0]))
        self.group1_idx = np.where((self.group_label==0)[0:-need_remove_num])[0]
        self.group2_idx = np.where((self.group_label==1)[0:-need_remove_num])[0]
        # self.group1_idx = np.where((self.df['race'].values == self.gp1_label)[0:-need_remove_num])[0]
        # self.group2_idx = np.where((self.df['race'].values == self.gp2_label)[0:-need_remove_num])[0]



class CImnistDataset(Dataset):
    def __init__(self, args, data, transform=None):
        self.device = args.device
        self.args = args
        self.transform = transform
        self.data = torch.FloatTensor(data.img)
        self.label_y = torch.tensor(data.eo_lbl, dtype=torch.long).to(self.device)

        self.y = np.zeros((data.eo_lbl.shape[0], int(data.eo_lbl.max()) + 1))
        self.y[np.arange(data.eo_lbl.shape[0]), data.eo_lbl.astype(int)] = 1
        self.y = torch.tensor(self.y, dtype=torch.long).to(self.device)

        self.group_indicator = data.att
        self.group1_idx = np.where(self.group_indicator==0)[0]
        self.group2_idx = np.where(self.group_indicator==1)[0]
        self.idxs = torch.tensor(range(self.data.shape[0]))

    def __getitem__(self, index):
        img, target, idx_array = self.data[index], self.y[index], self.idxs[index]
        batch_label_y = self.label_y[index]
        return idx_array, img, target, batch_label_y

    def __len__(self):
        return self.label_y.shape[0]



class CelebaDataset(Dataset):
    """Custom Dataset for loading CelebA face images"""
    def __init__(self, args, df, device, transform=None, label_category='Blond_Hair'):
        img_dir = parent_path + '/datasets/celeba/img_align_celeba/'
        self.df = df
        self.args = args
        self.img_dir = img_dir
        self.img_names = df.index.values
        # Black_Hair  Blond_Hair  Brown_Hair  Gray_Hair - too less
        # attribute1 = torch.tensor(df['Black_Hair'].values).unsqueeze(-1)
        # attribute2 = torch.tensor(df['Blond_Hair'].values).unsqueeze(-1)
        # attribute3 = torch.tensor(df['Brown_Hair'].values).unsqueeze(-1)
        # # attribute4 = torch.tensor(df['Gray_Hair'].values).unsqueeze(-1)
        # self.y = torch.cat([attribute1, attribute2, attribute3, attribute4], dim=-1).to(device)
        # assert (self.y.sum(dim=-1) == 1).all()
        # self.label_y =  torch.where(self.y)[1]
        self.y = F.one_hot(torch.tensor(df[label_category].values)).to(device)
        self.label_y = torch.tensor(df[label_category].values).to(device)
        self.group1_idx = np.where(df['Male'].values==0)[0] # female
        self.group2_idx = np.where(df['Male'].values==1)[0] # male
        self.group_label = df['Male'].values
        self.transform = transform
        self.device = device
        self.pretrain_data = None
        self.idxs = torch.tensor(range(self.y.shape[0]))

    def __getitem__(self, index):
        assert index < self.y.shape[0]
        if self.pretrain_data is not None:
            return self.idxs[index], self.pretrain_data[index], self.y[index], self.label_y[index], self.group_label[index]
        else:
            img = Image.open(os.path.join(self.img_dir, self.img_names[index]))
            if self.transform is not None:
                img = self.transform(img)
            return self.idxs[index], img, self.y[index], self.label_y[index], self.group_label[index]

    def __len__(self):
        return self.y.shape[0]

    def remove_last(self):
        num_data = self.y.shape[0]
        need_remove_num = num_data%self.args.tracker_bz
        if need_remove_num == 0:
            return

        self.y = self.y[0:-need_remove_num]
        self.label_y = self.label_y[0:-need_remove_num]
        self.pretrain_data = self.pretrain_data[0:-need_remove_num]
        self.idxs = torch.tensor(range(self.y.shape[0]))
        self.group1_idx = np.where((self.df['Male'].values==0)[0:-need_remove_num])[0] # female
        self.group2_idx = np.where((self.df['Male'].values==1)[0:-need_remove_num])[0] # male



class FairFaceDataset(Dataset):
    VALID_EXTENSION = ['*.jpg', '*.png']
    _repr_indent = 4
    def __init__(self, args, root, label_category, df, mean=None, std=None, train=True, device='cuda', attribute_mapping=None):
        self.args = args
        self.df = df
        assert label_category in ['age', 'gender', 'race']
        from torchvision import transforms
        transform = []
        if train:
            transform.append(transforms.RandomHorizontalFlip())
        transform.append(transforms.ToTensor())
        if mean is not None and std is not None:
            transform.append(transforms.Normalize(mean=mean, std=std))
        transform = transforms.Compose(transform)

        self.root = root
        self.transforms = transform
        self.paths = [os.path.join(root, name) for name in df.values[:, 0]]
        self.label_y = torch.tensor(np.array(df[label_category].values, dtype=np.int64))
        self.y = F.one_hot(self.label_y)
        self.idxs = torch.tensor(range(self.y.shape[0]))
        race_array = attribute_mapping['race']
        gp1_label = np.where(race_array == 'Black')[0][0]
        self.gp1_label = gp1_label
        gp2_label = np.where(race_array == 'White')[0][0]
        self.gp2_label = gp2_label
        self.group1_idx = np.where(df['race'].values == gp1_label)[0]
        self.group2_idx = np.where(df['race'].values == gp2_label)[0]
        self.label_y = self.label_y.to(device)
        self.y = self.y.to(device)
        self.pretrain_data = None

    def __len__(self) -> int:
        return self.y.shape[0]

    def __getitem__(self, idx):
        assert idx < self.y.shape[0]
        if self.pretrain_data is not None:
            return self.idxs[idx], self.pretrain_data[idx], self.y[idx], self.label_y[idx]
        else:
            img = Image.open(self.paths[idx]).convert("RGB")
            if self.transforms is not None:
                img = self.transforms(img)
            return self.idxs[idx], img, self.y[idx], self.label_y[idx]

    def __repr__(self) -> str:
        self.multi = 0
        head = "Dataset " + self.__class__.__name__
        body = ["multi: {}".format(self.multi)]
        body += ["Number of datapoints: {}".format(self.__len__())]
        if self.root is not None:
            body.append("Root location: {}".format(self.root))
        body += self.extra_repr().splitlines()
        if hasattr(self, "transforms") and self.transforms is not None:
            body += ["Transforms: {}".format(repr(self.transforms))]
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return '\n'.join(lines)

    def extra_repr(self) -> str:
        return ""

    def remove_last(self):
        num_data = self.y.shape[0]
        need_remove_num = num_data%self.args.tracker_bz
        if need_remove_num == 0:
            return
        self.y = self.y[0:-need_remove_num]
        self.label_y = self.label_y[0:-need_remove_num]
        self.pretrain_data = self.pretrain_data[0:-need_remove_num]
        self.idxs = torch.tensor(range(self.y.shape[0]))
        self.group1_idx = np.where((self.df['race'].values == self.gp1_label)[0:-need_remove_num])[0]
        self.group2_idx = np.where((self.df['race'].values == self.gp2_label)[0:-need_remove_num])[0]









if __name__ == '__main__':
    from sklearn.preprocessing import LabelEncoder
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data', type=str, default=parent_path+'/datasets/fairface', help='directory to data')
    parser.add_argument('--csv', type=str,  default=parent_path+'/datasets/fairface/fairface_label_train.csv',help='file to training annotation file')
    parser.add_argument('--vcsv', type=str, default=parent_path+'/datasets/fairface/fairface_label_val.csv',help='file to validation annotation file')
    parser.add_argument('--multi', action='store_true', default=False, help='multiple label')
    parser.add_argument('--evaluate', action='store_true', default=False, help='evaluate')
    parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN', help='Override mean pixel value of dataset')
    parser.add_argument('--std', type=float, nargs='+', default=None, metavar='STD', help='Override std deviation of of dataset')
    parser.add_argument('--no-use', action='store_true', default=False, help='no use')
    args = parser.parse_args()

    # train data
    df = pd.read_csv(args.csv)

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
        df[attribute] = le.transform(df[attribute].values) # df[attribute].apply(le.fit_transform)

    dataset = create_FairFace_dataset(root=args.data, label_category='age', df=df, train=True, attribute_mapping=attribute_mapping)
    print(dataset)