import argparse
import os
import time

import pandas as pd
import torch
import datetime
import numpy as np
from torch import optim, nn
from torch.utils.tensorboard import SummaryWriter

from data_loader import load_data
from models import MLP, CNN, LeNet, Autoencoder
from resnet import resnet18, resnet34, resnet50
from utils import set_seed, accuracy
import logging
from sklearn.metrics import roc_auc_score, recall_score, precision_score
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.abspath(os.path.join(dir_path, os.pardir))


def init_model(args, input_d=100):
    if args.model in ['logistic', 'mlp']:
        model = MLP(args)
    elif args.model == 'cnn':  # cnn
        model = CNN(args) # num params: 6582
    elif args.model == 'lenet':
        model = LeNet(args) # 61706
    elif args.model == 'autoencoder':
        if args.dataset == 'celebaimage':
            model = Autoencoder(args, input_d=args.raw_dim, layer=args.layer, hid=args.hidden_dim)
        elif args.dataset == 'compas' and args.dataset == 'celebatab':
            model = Autoencoder(args, input_d=args.raw_dim, layer=max(2, args.layer), hid=min(args.hidden_dim, 32))
        else:
            model = Autoencoder(args, input_d=args.raw_dim, layer=args.layer, hid=args.hidden_dim)
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

#
# def train(args, model, optimizer, train_dataloader, epoch):
#     criterion = nn.MSELoss()
#     model.train()
#     total_loss = 0
#     iteration = 0
#     for idx, x, onehot_y, label_y, group_label in tqdm(train_dataloader, delay=10):
#         model.zero_grad()
#         optimizer.zero_grad()
#         x = x.view(x.size(0), -1).float()
#         x = x.to(args.device)
#         logit, contra_loss, _ = model(x, group_label)
#         temp_loss = criterion(logit, x) + contra_loss
#         temp_loss.backward()
#         optimizer.step()
#         total_loss += temp_loss.item()
#         iteration += 1
#
#     print("[Train@Epoch:{}] Loss:{}".format(epoch, total_loss / iteration))
#
#     if 'tb' in args.log_type:
#         writer.add_scalar('Train/loss', total_loss, epoch)
#     return total_loss


def batch(iterable, batch_size=1):
    l = len(iterable)
    iterable.shuffle()
    for ndx in range(0, l, batch_size):
        yield iterable[ndx:min(ndx + batch_size, l)]


def train_with_fair_rebalance(args, model, optimizer, train_dataloader, epoch, group_ratio, flag):
    criterion = nn.MSELoss()
    model.train()
    total_loss = 0
    iteration = 0
    rec_loss = 0
    mse_accumulate = []
    for idx, x, onehot_y, label_y, group_label in batch(train_dataloader, args.train_batch_size):
        x = x.view(x.size(0), -1).float()
        x = x.to(args.device)

        model.zero_grad()
        optimizer.zero_grad()
        logit, contra_loss, old_weight = model(x, group_label)
        group_u_idx = torch.where(group_label == 0)[0]
        group_p_idx = torch.where(group_label == 1)[0]

        if len(group_u_idx) == 0:
            Lu_loss = 0
        else:
            # Lu_loss = criterion(logit[group_u_idx], x[group_u_idx])
            Lu_loss = torch.norm(logit[group_u_idx] - x[group_u_idx], p=2).sum()
        if len(group_p_idx) == 0:
            Lp_loss = 0
        else:
            # Lp_loss = criterion(logit[group_p_idx], x[group_p_idx])
            Lp_loss = torch.norm(logit[group_p_idx] - x[group_p_idx], p=2).sum()

        if args.weightmode == "new":
            Xu_avg = logit[group_u_idx].mean(dim=0, keepdim=True)
            Xp_avg = logit[group_p_idx].mean(dim=0, keepdim=True)
            oriXu_avg = x[group_u_idx].mean(dim=0, keepdim=True)
            oriXp_avg = x[group_p_idx].mean(dim=0, keepdim=True)
            if args.lossmode == 'lc1':
                Lu_0 = torch.norm(x[group_u_idx] - Xu_avg, p=2).sum()
                Lp_0 = torch.norm(x[group_p_idx] - Xp_avg, p=2).sum()
                weight = (Lu_0 - Lu_loss)/(Lu_0 - Lu_loss + Lp_0 - Lp_loss)
                weight = weight.detach()
            elif args.lossmode == 'lc2':
                Lu_0 = torch.norm(logit[group_u_idx] - Xu_avg, p=2).sum()
                Lp_0 = torch.norm(logit[group_p_idx] - Xp_avg, p=2).sum()
                weight = (Lu_0 - Lu_loss) / (Lu_0 - Lu_loss + Lp_0 - Lp_loss)
                weight = weight.detach()
            elif args.lossmode == 'lc3':
                Lu_0 = torch.norm(logit[group_u_idx], p=2).sum()
                Lp_0 = torch.norm(logit[group_p_idx], p=2).sum()
                weight = (Lu_0 - Lu_loss) / (Lu_0 - Lu_loss + Lp_0 - Lp_loss)
                weight = weight.detach()
            elif args.lossmode == 'lc4':
                Lu_0 = torch.norm(logit[group_u_idx] - oriXu_avg, p=2).sum()
                Lp_0 = torch.norm(logit[group_p_idx] - oriXp_avg, p=2).sum()
                weight = (Lu_0 - Lu_loss) / (Lu_0 - Lu_loss + Lp_0 - Lp_loss)
                weight = weight.detach()
            elif args.lossmode == 'lc5':
                Lu_0 = torch.norm(x[group_u_idx] - oriXu_avg, p=2).sum()
                Lp_0 = torch.norm(x[group_p_idx] - oriXp_avg, p=2).sum()
                weight = (Lu_0 - Lu_loss) / (Lu_0 - Lu_loss + Lp_0 - Lp_loss)
                weight = weight.detach()
            else:
                exit(0)

            # if not flag:
            #     weight = group_ratio
            if weight > 1 or weight < 0:
                print("[Train@Epoch:{}] Loss:{} Weight for the protected group:{}".format(epoch, rec_loss, weight.item()))
                weight = group_ratio
        elif args.weightmode == "old":
            weight = old_weight.detach()
        else:
            print("Wrong weight mode!")
            exit(0)
        rec_loss += ((1 - weight) * Lu_loss + weight * Lp_loss).item() / logit.shape[0]
        temp_loss = ((1 - weight) * Lu_loss + weight * Lp_loss) / logit.shape[0] + contra_loss
        # rec_loss += (weight * Lu_loss + (1 - weight) * Lp_loss).item()
        # temp_loss = weight * Lu_loss + (1 - weight) * Lp_loss + contra_loss
        temp_loss.backward()
        optimizer.step()
        total_loss += temp_loss.item()
        iteration += 1
        # gradient_norm = 0
        # gradient_norm = model.decoder[0].weight.grad.clone().norm(dim=1)
        # mse_accumulate.extend(gradient_norm)
        mse = torch.mean((x - logit) ** 2, dim=1).cpu().detach().numpy()
        mse_accumulate.extend(mse)

    if 'tb' in args.log_type:
        writer.add_scalar('Train/loss', total_loss, epoch)
    return rec_loss/iteration, torch.tensor(mse_accumulate).cpu(), weight


def test(args, model, test_dataloader, k=500):
    model.eval()
    mse_accumulate = []
    with torch.no_grad():
        for idx, x, onehot_y, label_y, group_label in batch(test_dataloader, args.train_batch_size):
            x = x.view(x.size(0), -1).float()
            x = x.to(args.device)
            reconstructed_images, _, _ = model(x, group_label)
            mse = torch.mean((x - reconstructed_images) ** 2, dim=1).cpu().detach().numpy()
            mse_accumulate.extend(mse)
    label_array = test_dataloader.label_y
    group_u_idx = test_dataloader.group1_idx
    group_p_idx = test_dataloader.group2_idx
    if args.dataset == 'mnistandusps':
        top_k = [1200, 1000, 800]
    elif args.dataset == 'mnistinvert':
        top_k = [500, 400, 300]
    elif args.dataset == 'celebatab':
        top_k = [5000, 4500, 4000]
    elif args.dataset == 'compas':
        top_k = [350, 300, 250]
    elif args.dataset == 'celebaimage':
        top_k = [2500, 2250, 2000]
    else:
        top_k = [5000, 4500, 4000]
    csv_data = []
    csv_recall = []
    iter = 0
    for k in top_k:
        print('When k=', k)
        iter += 1
        idx = torch.topk(torch.tensor(mse_accumulate), k=k)[1]
        pred = torch.zeros(label_array.shape[0]).to(args.device)
        pred[idx] = 1
        correct_pred = pred * label_array
        # Acc
        test_acc = accuracy(pred, label_array)
        group1_acc = accuracy(pred[test_dataloader.group1_idx], label_array[test_dataloader.group1_idx])
        group2_acc = accuracy(pred[test_dataloader.group2_idx], label_array[test_dataloader.group2_idx])
        recall = recall_score(label_array.cpu(), pred.cpu())
        print("The number of samples in Group 1 (truth):", label_array[group_u_idx].cpu().shape[0],
              "\t The number of Samples in Group 2 (truth):", label_array[group_p_idx].cpu().shape[0])
        print("The number of anomalies in Group 1 (truth):", label_array[group_u_idx].cpu().sum().item(),
              "\t The number of anomalies in Group 2 (truth):", label_array[group_p_idx].cpu().sum().item())
        print("The number of anomalies in Group 1 (pred):", pred[group_u_idx].cpu().sum().item(),
              "\t The number of anomalies in Group 2 (pred):", pred[group_p_idx].cpu().sum().item(),
              "\t The number of anomalies (pred):",  pred.cpu().sum().item())
        print("The number of correct anomalies in Group 1 (pred):", correct_pred[group_u_idx].cpu().sum().item(),
              "\t The number of correct anomalies in Group 2 (pred):", correct_pred[group_p_idx].cpu().sum().item(),
              "\t The number of correct anomalies (pred):",  correct_pred.cpu().sum().item())
        acc_diff = abs(group1_acc - group2_acc).item()
        prec_diff = abs(precision_score(label_array[group_u_idx].cpu(), pred[group_u_idx].cpu())
        - precision_score(label_array[group_p_idx].cpu(), pred[group_p_idx].cpu()))
        recall1 = recall_score(label_array[group_u_idx].cpu(), pred[group_u_idx].cpu())
        recall2 = recall_score(label_array[group_p_idx].cpu(), pred[group_p_idx].cpu())
        recall_diff = abs(recall1 - recall2)
        roc_auc = roc_auc_score(label_array.cpu().numpy(), mse_accumulate)
        print("AUCROC: {:.4f}, Recall:  {:.4f}, Accuracy:  {:.4f}, ACC Diff: {:.4f}, Precision Diff: {:.4f}, Recall Diff: {:.4f}".format(roc_auc, recall, test_acc.item(), acc_diff, prec_diff, recall_diff))
        if iter == 1:
            csv_data.extend([roc_auc, "{}({})".format(correct_pred[group_u_idx].cpu().sum().item(), pred[group_u_idx].cpu().sum().item()),
                             "{}({})".format(correct_pred[group_p_idx].cpu().sum().item(), pred[group_p_idx].cpu().sum().item())])
        csv_data.extend([acc_diff, prec_diff, recall_diff])
        csv_recall.append(recall)
    csv_recall.extend(csv_data)
    return csv_recall, top_k


def main(args, all_dataloader):
    # 1.init models
    start_time = time.time()
    args.num_class = 2
    patience = 0
    model = init_model(args, input_d=args.raw_dim)
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
    lowest_rec_loss = np.inf
    flag = False
    group_u_idx = torch.Tensor(all_dataloader.group1_idx).long().cpu()
    group_p_idx = torch.Tensor(all_dataloader.group2_idx).long().cpu()
    ### initialize the weight for two groups
    group_ratio = group_u_idx.shape[0] / (group_u_idx.shape[0] + group_p_idx.shape[0])
    # group_ratio = 0.5
    best_5recalls = []
    best_5aucrocs = []
    best_5recalldiffs = []
    last_results = []
    for epoch in range(1, args.epochs+1):
        # 2.1 train
        model.train()
        if epoch > 200:
            flag = True
        rec_loss, mse_accu, weight = train_with_fair_rebalance(args, model, optimizer, all_dataloader, epoch, group_ratio, flag)

        if epoch % 10 == 0:
            print("[Train@Epoch:{}] Loss:{} Weight for the protected group:{}".format(epoch, rec_loss, weight))
        # 2.2 test
        if epoch % args.test_interval == 0:
            print("testing...")
            if epoch == 20:
                print("Here!")
            model.eval()
            results, top_k = test(args, model, all_dataloader, k=args.k)
            results.insert(0, epoch)
            results.append(time.time() - start_time)
            if len(best_5recalls) < 5 or results[1] > min(best_5recalls, key=lambda x: x[1])[1]:
                best_5recalls.append(results)
                # Keep only the top 5 results sorted by accuracy
                best_5recalls.sort(key=lambda x: x[1], reverse=True)  # Sort descending by accuracy
                if len(best_5recalls) > 5:
                    best_5recalls.pop()
            if len(best_5aucrocs) < 5 or results[4] > min(best_5aucrocs, key=lambda x: x[4])[4]:
                best_5aucrocs.append(results)
                # Keep only the top 5 results sorted by accuracy
                best_5aucrocs.sort(key=lambda x: x[4], reverse=True)  # Sort descending by accuracy
                if len(best_5aucrocs) > 5:
                    best_5aucrocs.pop()
            if len(best_5recalldiffs) < 5 or results[9] < max(best_5recalls, key=lambda x: x[9])[9]:
                best_5recalldiffs.append(results)
                # Keep only the top 5 results sorted by accuracy
                best_5recalldiffs.sort(key=lambda x: x[9], reverse=False)  # Sort descending by accuracy
                if len(best_5recalldiffs) > 5:
                    best_5recalldiffs.pop()

        # early stopping criteria
        if lowest_rec_loss > rec_loss:
            lowest_rec_loss = rec_loss
            patience = 0
        else:
            patience += 1
        if patience > args.patience and epoch > 1000:
            print('Reconstruction error does not reduce for {} iterations. Early Stopping Criteria Satisfied....'.format(args.patience))
            break
    print("testing...")
    model.eval()
    last_results, top_k = test(args, model, all_dataloader, k=args.k)
    last_results.insert(0, epoch)
    last_results.append(time.time() - start_time)
    if args.save_csv:
        header = ['epoch', 'recall@{}'.format(top_k[0]), 'recall@{}'.format(top_k[1]), 'recall@{}'.format(top_k[2]), 'auc',
                  'group 1 (budget)', 'group 2 (budget)', 'acc diff ({})'.format(top_k[0]),
                  'precision diff ({})'.format(top_k[0]),	'recall diff({})'.format(top_k[0]),
                  'acc diff ({})'.format(top_k[1]),	'precision diff ({})'.format(top_k[1]),
                  'recall diff({})'.format(top_k[1]),	'acc diff ({})'.format(top_k[2]),
                  'precision diff ({})'.format(top_k[2]),	'recall diff({})'.format(top_k[2]),	'Time']
        # results.append(time.time() - start_time)
        results = np.array(best_5recalls + best_5aucrocs + best_5recalldiffs + [last_results])
        # print(results.shape)
        # exclude_columns = {0, 5, 6, 7, 16}  # Set of columns to exclude
        # columns_to_round = [i for i in range(results.shape[1]) if i not in exclude_columns]

        df = pd.DataFrame(np.array(results).reshape(len(results), -1), columns=header)
        df.to_csv('./results/{}_results_{}_layer{}_proj{}_gamma{}_loss{}_normal{}_seed{}.csv'.format(args.dataset, args.alpha, args.layer, args.projection, args.gamma, args.lossmode, args.normalize, args.seed), mode='a')
        print('The total running time is {:.4f}'.format(time.time() - start_time))
    print('The final weight is {:.4f}'.format(weight.item()))

def parsers_parser():
    parser = argparse.ArgumentParser()
    # dataset
    parser.add_argument('--seed', type=int, default=40)
    parser.add_argument('--log_type', type=str, default='')
    parser.add_argument('--outlier_ratio', type=float, default=0.9)

    parser.add_argument('--dataset', type=str, default='mnistinvert',
                                help='tabular: compas, adults, folktable, '
                                     # 'mnistandusps, mnistandusps_bin'
                                     'Image: celebA, fairface, clr_mnist')
    parser.add_argument('--label_category', type=str, default='gender',
                                help='fairface:age, gender, celebA:Blond_Hair, Brown_Hair')
    # epochs
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--test_interval', type=int, default=20)
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
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--patience', type=int, default=1000)
    parser.add_argument('--ratio', type=float, default=1.0)
    parser.add_argument('--layer', type=int, default=1)
    parser.add_argument('--weightmode', type=str, default="new")
    parser.add_argument('--lossmode', type=str, default="lc1")
    parser.add_argument('--projection', type=int, default=0)
    parser.add_argument('--normalize', type=int, default=1)
    args = parser.parse_args()
    args.save_csv = True
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
    return args


if __name__ == "__main__":
    args = parsers_parser()
    args.time = str(int(datetime.datetime.now().strftime("%Y%m%d%H%M%S")))
    set_seed(args.seed)
    if 'resnet' in args.model:
        args.tracker_bz = 50
    data_info = args.dataset
    ### load data
    # trn_ds, val_ds, test_ds, train_dataloader, valid_dataloader, test_dataloader, all_dataloader = load_data(args)
    trn_ds, val_ds, test_ds, all_ds = load_data(args)
    ### make logger
    writer = None
    if 'logging' in args.log_type:
        logging.basicConfig(level=logging.INFO, filename='{}/logs/{}.txt'.format(parent_path, args.title), filemode='w')
    if 'tb' in args.log_type:
        writer = SummaryWriter(comment=args.title)
    valid_result_cache, test_result_cache = [], []
    ### start main

    if args.dataset == 'compas':
        args.train_batch_size = 1024
        # args.alpha = 1
        args.hidden_dim = 32
        args.layer = 2
        args.projection = 50
        args.epochs = 6000
    elif args.dataset == 'celebatab':
        args.train_batch_size = 4096
        # args.alpha = 1
        args.hidden_dim = 32
        args.layer = 2
        args.projection = 50
        args.epochs = 5000
    elif args.dataset == 'mnistinvert':
        args.train_batch_size = 4096
        # args.alpha = 5.0
        args.hidden_dim = 128
        args.layer = 1
        args.projection = 50
        args.epochs = 3000
    elif args.dataset == 'mnistandusps':
        args.train_batch_size = 5000
        # args.alpha = 5.0
        args.hidden_dim = 128
        args.layer = 2
        args.projection = 50
        args.epochs = 5000
    else:
        raise NotImplementedError
    print(args)
    # for seed in [40, 41, 42]:
    for seed in [args.seed]:
        set_seed(seed)
        # args.seed = seed
        main(args, all_ds)
        print('\n\n\n')