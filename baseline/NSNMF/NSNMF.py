from homogeneous_model import NSNMF
import time
from utils import *
from data_loader import load_data
from sklearn.metrics import roc_auc_score, recall_score, precision_score
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.abspath(os.path.join(dir_path, os.pardir))

def main(args, dataloader):
    k = 10
    d = 128
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    activation = 'leakyrelu'
    print(device)

    # data.adj1 = data.adj1.to(device)
    # data.adj2 = data.adj2.to(device)
    # data.v1 = data.v1.to(device)
    # data.v2 = data.v2.to(device)
    # data.labels = data.labels.to(device)
    # feature = data.v1 + data.v2
    # feature = torch.cat([data.v1[:, :1000], data.v2[:, :1000]], dim=1)
    feature = torch.tensor(all_dataloader.dataset.data.reshape(all_dataloader.dataset.data.shape[0], -1)).to(device)
    label_array = all_dataloader.dataset.label_y
    group1_idx = all_dataloader.dataset.group1_idx
    group2_idx = all_dataloader.dataset.group2_idx

    model = NSNMF(feature.shape[1], k=k, n=feature.shape[0], alpha=args.alpha)
    sim = model.cosine_sim(feature, feature)
    model = model.to(device)
    # model.load_state_dict(torch.load('./best_GCN_model_{}_{}.pt'.format(k, run)).state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001, weight_decay=1e-4)
    model.epoch = args.epochs
    model.train()
    start_time = time.time()
    stopping_threshold = np.inf
    patience_iter = 0


    for epoch in range(1, model.epoch+1):
        optimizer.zero_grad()
        loss = model(sim, feature)
        loss.mean().backward()
        optimizer.step()
        if stopping_threshold > loss.mean().item():
            stopping_threshold = loss.mean().item()
            patience_iter = 0
        else:
            patience_iter += 1
        # print('cluster size:', model.hard_membership.sum(dim=0))
        if epoch % 200 == 0:
            print('Epoch [{}/{}], time: {:.4f}, Rec Loss: {:.4f}'.format(
                epoch, model.epoch, time.time() - start_time, loss.mean().item()))
            # start_time = time.time()
        if patience_iter >= args.patience:
            print('Early stopping criteria satisfied! Stopping training!')
            break
    outlier = 50
    score = torch.norm(feature - torch.matmul(model.W, model.H), 2, dim=1)
    if args.dataset == 'mnistandusps':
        top_k = [1200, 1000, 800]
    elif args.dataset == 'mnistinvert':
        top_k = [500, 400, 300]
    elif args.dataset == 'celebatab':
        top_k = [5000, 4500, 4000]
    elif args.dataset == 'compas':
        top_k = [350, 300, 250]
    else:
        top_k = [500, 400, 300]
    label_array = all_dataloader.dataset.label_y
    group1_idx = all_dataloader.dataset.group1_idx
    group2_idx = all_dataloader.dataset.group2_idx
    for k in top_k:
        print('When k=', k)
        idx = torch.topk(torch.tensor(score), k=k)[1]
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
        from sklearn.metrics import roc_auc_score
        roc_auc = roc_auc_score(label_array.cpu(), pred.cpu())
        print(
            "AUCROC: {:.4f}, Recall:  {:.4f}, Accuracy:  {:.4f}, ACC Diff: {:.4f}, Precision Diff: {:.4f}, Recall Diff: {:.4f}".format(
                roc_auc, recall, test_acc.item(), acc_diff, prec_diff, recall_diff))
        print('Time = ', time.time() - start_time)


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
    parser.add_argument('--epochs', type=int, default=1500)
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
    parser.add_argument('--ratio', type=float, default=1.0)
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

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    import argparse
    args = parsers_parser()
    set_seed(args.seed)
    trn_ds, val_ds, test_ds, train_dataloader, valid_dataloader, test_dataloader, all_dataloader = load_data(args)
    for run in range(3):
        np.random.seed(run)
        torch.manual_seed(run)
        print('')
        print(f'Run {run:02d}:')
        main(args, all_dataloader)
