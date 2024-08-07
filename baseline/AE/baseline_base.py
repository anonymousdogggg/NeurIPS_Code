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
from utils import set_seed, accuracy, save_final_result
import logging
from sklearn.metrics import roc_auc_score, recall_score

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
    total_loss  = 0

    for idx, x, onehot_y, label_y in tqdm(train_dataloader, delay=10):
        model.zero_grad()
        optimizer.zero_grad()
        x = x.view(x.size(0), -1).float()
        x = x.to(args.device)
        logit = model(x)
        temp_loss = criterion(logit, x)
        temp_loss.backward()
        optimizer.step()
        total_loss += temp_loss.item()
    print("[Train@Epoch:{}] Loss:{}".format(epoch, total_loss))

    if 'tb' in args.log_type:
        writer.add_scalar('Train/loss', total_loss, epoch)
    return total_loss


#
# def test(args, model, test_dataloader, epoch, type, save_logging=False):
#     model.eval()
#     mse_accumulate = []
#     with torch.no_grad():
#         for idx, x, onehot_y, label_y in test_dataloader:
#             x = x.view(x.size(0), -1).float()
#             x = x.to(args.device)
#             reconstructed_images = model(x)
#             mse = torch.mean((x - reconstructed_images) ** 2, dim=(1,2)).cpu().detach().numpy()
#             mse_accumulate.extend(mse)
#
#     threshold = np.mean(mse_accumulate) + 2 * np.std(mse_accumulate)
#     pred = torch.from_numpy(mse_accumulate > threshold).int().to(args.device)
#     label_array = test_dataloader.dataset.label_y
#
#     # Acc
#     test_acc = accuracy(pred, label_array)
#     group1_acc = accuracy(pred[test_dataloader.dataset.group1_idx], label_array[test_dataloader.dataset.group1_idx])
#     group2_acc = accuracy(pred[test_dataloader.dataset.group2_idx], label_array[test_dataloader.dataset.group2_idx])
#     print("Test accuracy:", test_acc, "\t Group 1 Acurracy:", group1_acc, "\t Group 2 Accuracy: ", group2_acc)

def test(args, model, test_dataloader, epoch, type, save_logging=False):
    model.eval()
    mse_accumulate = []
    with torch.no_grad():
        for idx, x, onehot_y, label_y in test_dataloader:
            x = x.view(x.size(0), -1).float()
            x = x.to(args.device)
            reconstructed_images = model(x)
            mse = torch.mean((x - reconstructed_images) ** 2, dim=(1, 2)).cpu().detach().numpy()
            # mse = torch.mean((x - reconstructed_images) ** 2, dim=1).cpu().detach().numpy()
            mse_accumulate.extend(mse)
    label_array = test_dataloader.dataset.label_y
    idx = torch.topk(torch.tensor(mse_accumulate), k=args.k)[1]
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
    print("Test accuracy:", test_acc, "\t Group 1 Accuracy:", group1_acc, "\t Group 2 Accuracy: ", group2_acc, "recall:", recall)
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
    model.smallest_diff_epoch = abs(group1_acc - group2_acc)
    model.smallest_diff = abs(group1_acc - group2_acc)


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

    # 2.training
    for epoch in range(1, args.epochs+1):
        # 2.1 test
        if epoch % args.test_interval == 0:
            print("testing...")
            model.eval()
            # test(args, model, test_dataloader, epoch, 'Test', True)
            test(args, model, all_dataloader, epoch, 'Test', True)

        # 2.2 train
        model.train()
        # train(args, model, optimizer, train_dataloader, epoch)
        train(args, model, optimizer, all_dataloader, epoch)

    if 'logging' in args.log_type:
        logging.info('[Test@Epoch:{}] Smallest Diff:{}.'.format(model.smallest_diff_epoch, model.smallest_diff))

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
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--test_interval', type=int, default=10)
    # batch size
    parser.add_argument('--train_batch_size', type=int, default=128)
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
    # trn_ds, val_ds, test_ds, train_dataloader, valid_dataloader, test_dataloader = load_data(args)
    trn_ds, val_ds, test_ds, train_dataloader, valid_dataloader, test_dataloader, all_dataloader = load_data(args)
    # args.title = '{}_{}_{}_Pre&Frz{}_{}-Lr{}_Norm{}_Decay{}_Epslr{}_{}_EpsInt{}_EpsNum{}_v{}_{}_{}'.format(eps_info, data_info, args.model,
    #                                     str(args.pretrained)+str(args.freeze_pretrain), args.optim, args.lr, args.norm_input, args.weight_decay,
    #                                     args.epslr, args.approx_type, args.eps_update_interval, args.topK, args.if_version, args.r, args.recursion_depth)

    ### make logger
    writer = None
    if 'logging' in args.log_type:
        logging.basicConfig(level=logging.INFO, filename='{}/logs/{}.txt'.format(parent_path, args.title), filemode='w')
    if 'tb' in args.log_type:
        writer = SummaryWriter(comment=args.title)
    valid_result_cache, test_result_cache = [], []
    ### start main
    main(args)