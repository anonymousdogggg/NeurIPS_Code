import argparse
import os
import pickle
import time
import torch
import datetime
import numpy as np
from torch import optim, nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data_loader import load_data
from models import MLP, CNN, LeNet, Autoencoder
from resnet import resnet18, resnet34, resnet50
from utils import set_seed, accuracy
import logging
from sklearn.metrics import roc_auc_score, recall_score, precision_score
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.abspath(os.path.join(dir_path, os.pardir))


def init_model(args):
    if args.model in ['logistic', 'mlp']:
        model = MLP(args)
    elif args.model == 'cnn':  # cnn
        model = CNN(args) # num params: 6582
    elif args.model == 'lenet':
        model = LeNet(args) # 61706
    elif args.model == 'autoencoder':
        model = Autoencoder(args)
    elif args.model == 'resnet18':
        model = resnet18(num_classes=args.num_class, pretrained=args.pretrained, freeze_pretrain=args.freeze_pretrain)
    elif args.model == 'resnet34':
        model = resnet34(num_classes=args.num_class, pretrained=args.pretrained, freeze_pretrain=args.freeze_pretrain)
    elif args.model == 'resnet50':
        model = resnet50(num_classes=args.num_class, pretrained=args.pretrained, freeze_pretrain=args.freeze_pretrain)
    else:
        model = None
        exit(102)
    return model



def train(args, model, optimizer, train_dataloader, epoch):
    criterion = nn.MSELoss()
    model.train()
    total_loss = 0
    iteration = 0
    for idx, x, onehot_y, label_y, group_label in tqdm(train_dataloader, delay=10):
        model.zero_grad()
        optimizer.zero_grad()
        x = x.view(x.size(0), -1).float()
        x = x.to(args.device)
        logit, contra_loss, weight = model(x, group_label)
        temp_loss = criterion(logit, x) + contra_loss
        temp_loss.backward()
        optimizer.step()
        total_loss += temp_loss.item()
        iteration += 1
        # if iteration % 25 == 0:
    print("[Train@Epoch:{}] Loss:{}".format(epoch, total_loss/iteration))

    if 'tb' in args.log_type:
        writer.add_scalar('Train/loss', total_loss, epoch)
    return total_loss


def train_new(args, model, h_prime, optimizer, optimizer_prime, train_dataloader, epoch):
    criterion = nn.MSELoss()
    dst_criterion = nn.MSELoss()  # TODO
    model.train()
    total_loss = 0
    iteration = 0
    for idx, x, onehot_y, label_y, group_label in tqdm(train_dataloader, delay=10):
        model.zero_grad()
        optimizer_prime.zero_grad()
        x = x.view(x.size(0), -1).float()
        x = x.to(args.device)
        emb, hx = model.getMinmaxEmb(x)
        hpx = h_prime(emb)
        group1_idx = torch.where(group_label == 0)[0]
        group2_idx = torch.where(group_label == 1)[0]
        # how to choose a proper distance function? Does criterion make sense?
        dst = dst_criterion(hx[group1_idx], hpx[group1_idx]) - criterion(hx[group2_idx], hpx[group2_idx])
        loss_hp = -dst**2  # Negative for maximization
        loss_hp.backward()
        optimizer_prime.step()

        optimizer.zero_grad()
        emb, hx = model.getMinmaxEmb(x)
        hpx = h_prime(emb)
        logit, _, _ = model(x, group_label)
        dst = dst_criterion(hx[group1_idx], hpx[group1_idx]) - criterion(hx[group2_idx], hpx[group2_idx])
        temp_loss = criterion(logit, x) + args. minmaxweight * dst**2
        temp_loss.backward()
        optimizer.step()
        total_loss += temp_loss.item()
        iteration += 1
        # if iteration % 25 == 0:
    print("[Train@Epoch:{}] Loss:{}".format(epoch, total_loss/iteration))

    if 'tb' in args.log_type:
        writer.add_scalar('Train/loss', total_loss, epoch)
    return total_loss


def train_with_fair_rebalance(args, model, optimizer, train_dataloader, epoch):
    criterion = nn.MSELoss()
    model.train()
    total_loss = 0
    iteration = 0
    weight_mean = 0
    for idx, x, onehot_y, label_y, group_label in tqdm(train_dataloader, delay=10):
        model.zero_grad()
        optimizer.zero_grad()
        x = x.view(x.size(0), -1).float()
        x = x.to(args.device)
        group1_idx = torch.where(group_label == 0)[0]
        group2_idx = torch.where(group_label == 1)[0]
        logit, contra_loss, weight = model(x, group_label)
        if len(group1_idx) == 0:
            group1_loss = 0
        else:
            group1_loss = criterion(logit[group1_idx], x[group1_idx])
        if len(group2_idx) == 0:
            group2_loss = 0
        else:
            group2_loss = criterion(logit[group2_idx], x[group2_idx])
        # weight = len(group1_idx)/(len(group1_idx) + len(group2_idx))
        # weight = 0.55
        if args.method == 'base':
            weight = 0.5
        weight_mean += weight
        temp_loss = (1-weight) * group1_loss + weight * group2_loss + contra_loss
        temp_loss.backward()
        optimizer.step()
        total_loss += temp_loss.item()
        iteration += 1
        # if iteration % 25 == 0:
    print("[Train@Epoch:{}] Loss:{} Weight:{} ".format(epoch, total_loss/iteration, weight_mean/iteration))

    if 'tb' in args.log_type:
        writer.add_scalar('Train/loss', total_loss, epoch)
    return total_loss


def test(args, model, test_dataloader, epoch, type, save_logging=False, k=500):
    model.eval()
    mse_accumulate = []
    with torch.no_grad():
        for idx, x, onehot_y, label_y, group_label in test_dataloader:
            x = x.view(x.size(0), -1).float()
            x = x.to(args.device)
            reconstructed_images, _, _ = model(x)
            # mse = torch.mean((x - reconstructed_images) ** 2, dim=(1, 2)).cpu().detach().numpy()
            mse = torch.mean((x - reconstructed_images) ** 2, dim=1).cpu().detach().numpy()
            mse_accumulate.extend(mse)
    label_array = test_dataloader.dataset.label_y
    print("Label_y shape:", len(label_array), "MSE shape:", len(mse_accumulate))
    group1_idx = test_dataloader.dataset.group1_idx
    group2_idx = test_dataloader.dataset.group2_idx
    if args.dataset == 'mnistandusps':
        top_k = [1200, 1000, 800]
    elif args.dataset == 'mnistinvert':
        top_k = [500, 400, 300]
    elif args.dataset == 'celebatab':
        top_k = [4500, 4000, 3500]
    elif args.dataset == 'compas':
        top_k = [400, 350, 300]
    else:
        top_k = [500, 400, 300]
    for k in top_k:
        print('When k=', k)

        idx = torch.topk(torch.tensor(mse_accumulate), k=k)[1]
        # threshold = np.mean(mse_accumulate) + 2 * np.std(mse_accumulate)
        # pred = torch.from_numpy(mse_accumulate > threshold).int().to(args.device)
        # pred = torch.from_numpy(mse_accumulate > threshold).int().to(args.device)
        pred = torch.zeros(label_array.shape[0]).to(args.device)
        pred[idx] = 1
        correct_pred = pred * label_array
        # Acc
        test_acc = accuracy(pred, label_array)
        group1_acc = accuracy(pred[test_dataloader.dataset.group1_idx], label_array[test_dataloader.dataset.group1_idx])
        group2_acc = accuracy(pred[test_dataloader.dataset.group2_idx], label_array[test_dataloader.dataset.group2_idx])
        # auc_roc = roc_auc_score(label_array[test_dataloader.dataset.group2_idx].cpu(), pred[test_dataloader.dataset.group2_idx].cpu())
        # auc_roc = roc_auc_score(label_array[test_dataloader.dataset.group2_idx].cpu(), mse_accumulate[test_dataloader.dataset.group2_idx].cpu())
        recall = recall_score(label_array.cpu(), pred.cpu())
        print("Test accuracy:", test_acc.item(), "\t Group 1 Accuracy:", group1_acc.item(), "\t Group 2 Accuracy: ", group2_acc.item(), "recall:", recall)
        # print("Test accuracy:", test_acc, "\t Group 1 Accuracy:", group1_acc, "\t Group 2 Accuracy: ", group2_acc, "AUC:", auc_roc)
        print("The number of samples in Group 1 (truth):", label_array[test_dataloader.dataset.group1_idx].cpu().shape[0],
              "\t The number of Samples in Group 2 (truth):", label_array[test_dataloader.dataset.group2_idx].cpu().shape[0])
        print("The number of anomalies in Group 1 (truth):", label_array[test_dataloader.dataset.group1_idx].cpu().sum().item(),
              "\t The number of anomalies in Group 2 (truth):", label_array[test_dataloader.dataset.group2_idx].cpu().sum().item())
        print("The number of anomalies in Group 1 (pred):", pred[test_dataloader.dataset.group1_idx].cpu().sum().item(),
              "\t The number of anomalies in Group 2 (pred):", pred[test_dataloader.dataset.group2_idx].cpu().sum().item(),
              "\t The number of anomalies (pred):",  pred.cpu().sum().item())
        print("The number of correct anomalies in Group 1 (pred):", correct_pred[test_dataloader.dataset.group1_idx].cpu().sum().item(),
              "\t The number of correct anomalies in Group 2 (pred):", correct_pred[test_dataloader.dataset.group2_idx].cpu().sum().item(),
              "\t The number of correct anomalies (pred):",  correct_pred.cpu().sum().item())
        acc_diff = abs(group1_acc - group2_acc)
        prec_diff = abs(precision_score(label_array[group1_idx].cpu(), pred[group1_idx].cpu())
                        - precision_score(label_array[group2_idx].cpu(), pred[group2_idx].cpu()))
        recall1 = recall_score(label_array[group1_idx].cpu(), pred[group1_idx].cpu())
        recall2 = recall_score(label_array[group2_idx].cpu(), pred[group2_idx].cpu())
        recall_diff = abs(recall1 - recall2)
        print("ACC Diff:", acc_diff, "Precision Diff:", prec_diff, 'Recall Diff:', recall_diff)


def main(args):
    # 1.init models
    args.num_class = 2
    model = init_model(args)
    model = model.to(args.device)
    if args.optim == 'Adam':
        optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr,
                               weight_decay=args.weight_decay)

    else:
        optimizer = optim.SGD([p for p in model.parameters() if p.requires_grad], lr=args.lr,
                               weight_decay=args.weight_decay)
    ##
    print('Begin training')
    # 2.training
    for epoch in range(1, args.epochs+1):
        # 2.1 test
        if epoch % args.test_interval == 0:
            print("testing...")
            model.eval()
            # test(args, model, test_dataloader, epoch, 'Test', True)
            test(args, model, all_dataloader, epoch, 'Test', True, k=args.k)

        # 2.2 train
        model.train()
        if args.alpha == 0:
            train(args, model, optimizer, all_dataloader, epoch)
        else:
            train_with_fair_rebalance(args, model, optimizer, all_dataloader, epoch)

    if 'logging' in args.log_type:
        logging.info('[Test@Epoch:{}] Smallest Diff:{}.'.format(model.smallest_diff_epoch, model.smallest_diff))
        print('[Test@Epoch:{}] Smallest Diff:{}.'.format(model.smallest_diff_epoch, model.smallest_diff))
    # save_final_result(args, valid_result_cache, test_result_cache)


def parsers_parser():
    parser = argparse.ArgumentParser()
    # dataset
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--part_data', type=int, default=0)
    parser.add_argument('--othergroup', type=int, default=0)
    parser.add_argument('--log_type', type=str, default='')
    parser.add_argument('--resplit', type=int, default=0, help='if resplit=0, then use original split.')
    parser.add_argument('--valid_ratio', type=float, default=0.2)
    parser.add_argument('--test_ratio', type=float, default=0.2)
    parser.add_argument('--outlier_ratio', type=float, default=0.9)

    parser.add_argument('--dataset', type=str, default='mnistandusps',
                                help='tabular: compas, adults, folktable, '
                                     # 'mnistandusps, mnistandusps_bin'
                                     'Image: celebA, fairface, clr_mnist')
    parser.add_argument('--label_category', type=str, default='gender',
                                help='fairface:age, gender, celebA:Blond_Hair, Brown_Hair')
    # epochs
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--epochs_stage1', type=int, default=100)
    parser.add_argument('--test_interval', type=int, default=10)
    # batch size
    parser.add_argument('--train_batch_size', type=int, default=256)
    parser.add_argument('--valid_batch_size', type=int, default=64)
    parser.add_argument('--test_batch_size', type=int, default=64)
    parser.add_argument('--tracker_bz', type=int, default=128)
    # model
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--reinit', type=int, default=1)
    parser.add_argument('--pretrained', type=int, default=0)
    parser.add_argument('--model', type=str, default='autoencoder', help='logistic, mlp, cnn, '
                                                              'resnet18, resnet34, resnet50'
                                                              'linear_resnet18, linear_resnet34, linear_resnet50') # TODO: check this
    parser.add_argument('--norm_input', type=int, default=0)
    parser.add_argument('--hidden_dim',  type=int, default=128)
    parser.add_argument('--l2_lambda', type=float, default=1)
    parser.add_argument('--optim', type=str, default='Adam', choices=['SGD', 'Adam', 'GD'])
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--method', type=str, default='FairCAD')
    parser.add_argument('--hidden-dim', type=int, default=128)


    # autoencoder parameters
    parser.add_argument('--gamma', type=float, default=0)
    parser.add_argument('--alpha', type=float, default=0)

    # minmaxparameter
    parser.add_argument('--minmaxweight', type=float, default=1.0)
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
    return args


if __name__ == "__main__":
    args = parsers_parser()
    args.time = str(int(datetime.datetime.now().strftime("%Y%m%d%H%M%S")))
    set_seed(args.seed)
    if 'resnet' in args.model:
        args.tracker_bz = 50

    ### make title

    data_info = args.dataset
    valid_info = str(args.valid_ratio) if args.resplit else 'default'



    ### load data
    trn_ds, val_ds, test_ds, train_dataloader, valid_dataloader, test_dataloader, all_dataloader = load_data(args)

    ### make logger
    writer = None
    if 'logging' in args.log_type:
        logging.basicConfig(level=logging.INFO, filename='{}/logs/{}.txt'.format(parent_path, args.title), filemode='w')
    if 'tb' in args.log_type:
        writer = SummaryWriter(comment=args.title)
    valid_result_cache, test_result_cache = [], []

    main(args)