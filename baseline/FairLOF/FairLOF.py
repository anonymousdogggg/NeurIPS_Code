"""
Implementation for competitive method FairLOF: https://arxiv.org/abs/2005.09900

Date: 12/2020
"""

from __future__ import division
import sys, os
import time
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from collections import defaultdict
import random
from data_loader import load_data
import argparse
from sklearn.metrics import recall_score, precision_score
from utils import set_seed, accuracy

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)


cuda0 = torch.device('cuda:0')
cuda1 = torch.device('cuda:1')
cuda2 = torch.device('cuda:2')
cuda3 = torch.device('cuda:3')


def distance_euclidean(middle, instance_list):
    """
    Split the data into two parts and have four matrices on four GPUs to save memory
    Args:
        middle: the middle index
        instance_list: the list needed to be split

    Returns: 4 matrices, altogether create a distance matrix between each instance in the original instance list

    """
    first_half = instance_list[:middle]
    second_half = instance_list[middle:]
    matrix11 = torch.cdist(torch.tensor(first_half).to(cuda0), torch.tensor(first_half).to(cuda0))
    matrix12 = torch.cdist(torch.tensor(first_half).to(cuda1), torch.tensor(second_half).to(cuda1))
    matrix21 = torch.cdist(torch.tensor(second_half).to(cuda2), torch.tensor(first_half).to(cuda2))
    matrix22 = torch.cdist(torch.tensor(second_half).to(cuda3), torch.tensor(second_half).to(cuda3))
    return matrix11, matrix12, matrix21, matrix22


class FairLOF:
    def __init__(self, db, k, instances, sensitive_attribute_group, distance_function=distance_euclidean):
        self.db = db
        self.k = k
        self.instances = instances
        self.sag = sensitive_attribute_group
        self.distance_function = distance_function
        self.middle = 0
        self.distance_matrices = NotImplemented
        self.LOF_kdistance = defaultdict(float)
        self.kdistance_values = defaultdict(float)
        self.kneighbors = defaultdict(list)
        self.density_map = defaultdict(float)
        self.Divs = defaultdict(float)
        self.Dxs = [0] * len(set(sensitive_attribute_group))
        self.Ws = 1
        self.DDxs = -1
        self.t = min(500, int(0.05 * len(instances)))
        self.set_up()

    def set_up(self):
        """ Caching neighbors for all instances """
        print('Caching...')
        """Caching Dxs and Ws"""
        for i in self.sag:
            self.Dxs[i] += 1
        for i in range(len(self.Dxs)):
            self.Dxs[i] /= len(self.sag)
        Dxs_array = np.array(self.Dxs)
        max_index = np.argmax(Dxs_array)
        Drlof = np.load(f'../{self.db}/Ws.npy')
        if Dxs_array[max_index] > Drlof[max_index]:
            self.DDxs = 1
        self.Ws += abs(Dxs_array[max_index] - Drlof[max_index])

        # -----Caching distance, k_distance, k_neighbors----- #
        middle = int(len(self.instances) / 2)
        self.middle = middle
        self.distance_matrices = self.distance_function(middle, self.instances)

        for i in range(len(self.instances)):
            current_sag = self.sag[i]
            if i < middle:
                if i < middle / 2:
                    temp_cuda = cuda0
                else:
                    temp_cuda = cuda1
                distance_list = torch.cat(
                    (self.distance_matrices[0][i, :].to(temp_cuda), self.distance_matrices[1][i, :].to(temp_cuda)), -1)
            else:
                if i < middle * 1.5:
                    temp_cuda = cuda2
                else:
                    temp_cuda = cuda3
                distance_list = torch.cat((self.distance_matrices[2][i - middle, :].to(temp_cuda),
                                           self.distance_matrices[3][i - middle, :].to(temp_cuda)), -1)
            all_distances, all_distances_index = torch.sort(distance_list, descending=False)
            count = 0
            div_count = 0
            current_neighbors = []
            last_distance = NotImplemented

            for j in range(len(all_distances)):
                if j == 0:
                    continue
                cur_distance = all_distances[j]
                if last_distance is NotImplemented or cur_distance != last_distance:
                    count += 1
                last_distance = cur_distance
                current_neighbor_index = int(all_distances_index[j])
                if current_sag != self.sag[current_neighbor_index]:
                    div_count += 1
                current_neighbors.append(current_neighbor_index)
                if count == self.k:
                    self.kneighbors[i] = current_neighbors
                    self.Divs[i] = div_count / len(current_neighbors)
                    self.LOF_kdistance[i] = last_distance
                    break
        print('Finished caching!')

    def get_k_distance(self, lbda):
        for i in range(len(self.instances)):
            current_sag = self.sag[i]
            self.kdistance_values[i] = self.LOF_kdistance[i] * (1 - lbda * self.DDxs * self.Ws
                                                                * self.Dxs[current_sag] * self.Divs[i])

    def get_value_neighbors(self, index):
        return self.kdistance_values[index], self.kneighbors[index]

    def local_outlier_factor(self, index, lbda):
        """
        Calculate local outlier factor, which is essentially the outlier score
        """
        (k_distance_value, neighbours) = self.get_value_neighbors(index)
        if self.density_map[index] == 0.0:
            instance_lrd = self.local_reachability_density(index, lbda)
            self.density_map[index] = instance_lrd
        else:
            instance_lrd = self.density_map[index]

        lrd_ratios_array = [0] * len(neighbours)
        for i, neighbour_index in enumerate(neighbours):
            # instances_without_instance = set(self.instances)
            # instances_without_instance.discard(neighbour)
            if self.density_map[neighbour_index] == 0.0:
                neighbour_lrd = self.local_reachability_density(neighbour_index, lbda)
                self.density_map[neighbour_index] = neighbour_lrd
            else:
                neighbour_lrd = self.density_map[neighbour_index]
            lrd_ratios_array[i] = neighbour_lrd / instance_lrd
        return sum(lrd_ratios_array) / len(neighbours)

    def local_reachability_density(self, index, lbda):
        """
          Calculate local reachability density, which is required for local outlier factor calculation
        """
        (k_distance_value, neighbours) = self.get_value_neighbors(index)
        reachability_distances_array = [0] * len(neighbours)  # n.zeros(len(neighbours))
        for i, neighbour in enumerate(neighbours):
            temp_index = index
            temp_neighbor = neighbour
            if index < self.middle:
                if neighbour < self.middle:
                    temp_cuda = cuda0
                    case = 0
                else:
                    temp_cuda = cuda1
                    case = 1
                    temp_neighbor -= self.middle
            else:
                temp_index -= self.middle
                if neighbour < self.middle:
                    temp_cuda = cuda2
                    case = 2
                else:
                    temp_cuda = cuda3
                    case = 3
                    temp_neighbor -= self.middle

            Inequality_check = 1
            if self.sag[index] == self.sag[neighbour]:
                Inequality_check = 0
            reachability_distances_array[i] = \
                float(torch.max(self.kdistance_values[neighbour].to(temp_cuda),
                                self.distance_matrices[case][temp_index][temp_neighbor].to(temp_cuda))
                      * (1 - lbda * self.DDxs * self.Ws * self.Dxs[self.sag[index]] * Inequality_check))
        return len(neighbours) / sum(reachability_distances_array)


def run(db, k, t, instances, sensitive_attribute_group, LOF_scores):
    guide_point = 0.9
    l = FairLOF(db, k, instances, sensitive_attribute_group)

    last_scores = NotImplemented
    bestscores = NotImplemented
    last_jacc = -1
    for lbda in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        scores = []
        l.get_k_distance(lbda)
        for i in range(len(instances)):
            score = l.local_outlier_factor(i, lbda)
            scores.append(float(score))
        scores = np.array(scores)
        top_t_fairLOF = np.argsort(-scores)[:t]
    #     top_t_LOF = np.argsort(-LOF_scores)[:t]
    #     jacc = len(np.intersect1d(top_t_LOF, top_t_fairLOF)) / len(np.union1d(top_t_LOF, top_t_fairLOF))
    #     print(f'Jacc for lbda = {lbda} is:    {jacc}')
    #     if abs(guide_point - jacc) > abs(guide_point - last_jacc):
    #         bestscores = last_scores
    #         break
    #     else:
    #         last_scores = scores
    #         last_jacc = jacc
    #
    # if bestscores is NotImplemented:
    #     bestscores = last_scores
    # # np.save(f'../{db}/FairLOF.npy', bestscores)
    return scores


def parsers_parser():
    parser = argparse.ArgumentParser()
    # dataset
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log_type', type=str, default='')
    parser.add_argument('--outlier_ratio', type=float, default=0.9)

    parser.add_argument('--dataset', type=str, default='mnistandusps',
                                help='tabular: compas, adults, folktable, '
                                     # 'mnistandusps, mnistandusps_bin'
                                     'Image: celebA, fairface, clr_mnist')
    parser.add_argument('--label_category', type=str, default='gender',
                                help='fairface:age, gender, celebA:Blond_Hair, Brown_Hair')
    # epochs
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--test_interval', type=int, default=10)
    # batch size
    parser.add_argument('--train_batch_size', type=int, default=256)
    parser.add_argument('--valid_batch_size', type=int, default=64)
    parser.add_argument('--test_batch_size', type=int, default=64)
    # model
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--model', type=str, default='autoencoder', help='logistic, mlp, cnn, '
                                                              'resnet18, resnet34, resnet50'
                                                              'linear_resnet18, linear_resnet34, linear_resnet50') # TODO: check this
    parser.add_argument('--hidden_dim',  type=int, default=128)
    parser.add_argument('--optim', type=str, default='Adam', choices=['SGD', 'Adam', 'GD'])
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--beta', type=float, default=1)
    parser.add_argument('--gamma', type=float, default=1)
    parser.add_argument('--patience', type=int, default=80)
    args = parser.parse_args()
    # save setting config into args
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.dataset == 'celebA':
        args.label_category = 'Blond_Hair'
    if args.dataset == 'fairface':
        args.label_category = 'gender'
    if (args.dataset == 'celebA' or args.dataset == 'fairface') and args.model == 'mlp':
        args.pretrained = 1
        args.freeze_pretrain = 1
    # set flag
    args.already_eps = 0
    device = args.device
    return args


def set_seed(seed):
    print(f"setting seed to {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.set_deterministic(True)
    torch.backends.cudnn.benchmark = False


def main():
    # db = sys.argv[1]
    db = 'dataset'
    args = parsers_parser()
    # X_arrays = np.load(f'../../datasets/{db.upper()}_X.npy')
    # X_norm = []
    # for i in X_arrays:
    #     X_norm.append(i)
    # Y_arrays = np.load(f'../../datasets/{db.upper()}_Y.npy')
    # Y = []
    # for i in Y_arrays:
    #     Y.append(i)
    #
    # sensitive_attribute_group = np.load(f'../{db}/attribute.npy', allow_pickle=True)
    trn_ds, val_ds, test_ds, train_dataloader, valid_dataloader, test_dataloader, all_dataloader = load_data(args)
    X_norm = all_dataloader.dataset.data.reshape(all_dataloader.dataset.data.shape[0], -1)
    Y = all_dataloader.dataset.label_y
    sensitive_attribute_group = all_dataloader.dataset.group_label
    input = np.reshape(sensitive_attribute_group, (-1, 1))
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(input)
    one_hot = enc.transform(input).toarray()
    sensitive_attribute_group = np.argmax(one_hot, axis=1)

    t = min(500, int(0.05 * len(X_norm)))
    # LOF_scores = np.load(f'../{db}/mylof.npy')
    # starttime = time.time()
    LOF_scores = None
    scores = run(db, 5, t, X_norm, sensitive_attribute_group, LOF_scores)

    if args.dataset == 'mnistandusps':
        top_k = [1200, 1000, 800]
    elif args.dataset == 'mnistinvert':
        top_k = [500, 400, 300]
    elif args.dataset == 'celebatab':
        top_k = [4500, 4000, 3500]
    elif args.dataset == 'compas':
        top_k = [350, 300, 250]
    else:
        top_k = [500, 400, 300]
    label_array = all_dataloader.dataset.label_y
    group1_idx = all_dataloader.dataset.group1_idx
    group2_idx = all_dataloader.dataset.group2_idx
    for k in top_k:
        print('When k=', k)
        idx = torch.topk(torch.tensor(scores), k=k)[1]
        pred = torch.zeros(label_array.shape[0]).to(args.device)
        pred[idx] = 1
        correct_pred = pred * label_array
        # Acc
        test_acc = accuracy(pred, label_array)
        group1_acc = accuracy(pred[all_dataloader.dataset.group1_idx],
                              label_array[all_dataloader.dataset.group1_idx])
        group2_acc = accuracy(pred[all_dataloader.dataset.group2_idx],
                              label_array[all_dataloader.dataset.group2_idx])
        # auc_roc = roc_auc_score(label_array[all_dataloader.dataset.group2_idx].cpu(), pred[all_dataloader.dataset.group2_idx].cpu())
        # auc_roc = roc_auc_score(label_array[all_dataloader.dataset.group2_idx].cpu(), mse_accumulate[all_dataloader.dataset.group2_idx].cpu())
        recall = recall_score(label_array.cpu(), pred.cpu())
        # print("Test accuracy:", test_acc, "\t Group 1 Accuracy:", group1_acc, "\t Group 2 Accuracy: ", group2_acc, "recall:", recall)

        # print("Test accuracy:", test_acc, "\t Group 1 Accuracy:", group1_acc, "\t Group 2 Accuracy: ", group2_acc, "AUC:", auc_roc)
        print("The number of samples in Group 1 (truth):", label_array[group1_idx].cpu().shape[0],
              "\t The number of Samples in Group 2 (truth):", label_array[group2_idx].cpu().shape[0])
        print("The number of anomalies in Group 1 (truth):", label_array[group1_idx].cpu().sum().item(),
              "\t The number of anomalies in Group 2 (truth):", label_array[group2_idx].cpu().sum().item())
        print("The number of anomalies in Group 1 (pred):", pred[group1_idx].cpu().sum().item(),
              "\t The number of anomalies in Group 2 (pred):", pred[group2_idx].cpu().sum().item(),
              "\t The number of anomalies (pred):", pred.cpu().sum().item())
        print("The number of correct anomalies in Group 1 (pred):", correct_pred[group1_idx].cpu().sum().item(),
              "\t The number of correct anomalies in Group 2 (pred):", correct_pred[group2_idx].cpu().sum().item(),
              "\t The number of correct anomalies (pred):", correct_pred.cpu().sum().item())
        acc_diff = abs(group1_acc - group2_acc).item()
        prec_diff = abs(precision_score(label_array[group1_idx].cpu(), pred[group1_idx].cpu())
                        - precision_score(label_array[group2_idx].cpu(), pred[group2_idx].cpu()))
        recall1 = recall_score(label_array[group1_idx].cpu(), pred[group1_idx].cpu())
        recall2 = recall_score(label_array[group2_idx].cpu(), pred[group2_idx].cpu())
        recall_diff = abs(recall1 - recall2)
        print(
            "Recall:  {:.4f}, Accuracy:  {:.4f}, ACC Diff: {:.4f}, Precision Diff: {:.4f}, Recall Diff: {:.4f}".format(
                recall, test_acc.item(), acc_diff, prec_diff, recall_diff))


if __name__ == '__main__':
    main()
