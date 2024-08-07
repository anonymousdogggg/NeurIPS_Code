import torch
from tqdm import tqdm
import torch.optim as optim
from DataSet.DataLoader import get_dataloader
from DataSet.new_dataloader import load_data
from Model.my_Model import CSVmodel
from Model.Loss import LossFunction
from Model.Score import ScoreFunction
from sklearn.metrics import recall_score, precision_score, roc_auc_score
from utils import aucPerformance, get_logger, F1Performance, accuracy

class Trainer(object):
    def __init__(self, args, model_config: dict):
        # self.run = run
        self.dataset = args.dataset
        self.ratio = args.ratio
        self.seed = model_config['random_seed']
        self.sche_gamma = model_config['sche_gamma']
        self.device = model_config['device']
        self.learning_rate = model_config['learning_rate']
        self.model = CSVmodel(model_config).to(self.device)
        self.loss_fuc = LossFunction(model_config).to(self.device)
        self.score_func = ScoreFunction(model_config).to(self.device)
        # self.train_loader, self.test_loader = get_dataloader(model_config)
        _, _, _, _, _, _, self.train_loader = load_data(args)
        self.test_loader = self.train_loader
        model_config['data_dim'] = args.raw_dim

    def training(self, epochs):
        train_logger = get_logger('train_log.log')
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.sche_gamma)
        self.model.train()
        print("Training Start.")
        min_loss = 100
        for epoch in range(epochs):
            for idx, x_input, onehot_y, y_label, group_label in tqdm(self.train_loader, delay=10):
                x_input = x_input.to(self.device)
                x_pred, z, masks = self.model(x_input)
                loss, mse, divloss = self.loss_fuc(x_input, x_pred, masks)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()
            info = 'Epoch:[{}]\t loss={:.4f}\t mse={:.4f}\t divloss={:.4f}\t'
            train_logger.info(info.format(epoch,loss.cpu(),mse.cpu(),divloss.cpu()))
            if loss < min_loss:
                torch.save(self.model, "model_seed{}.pth".format(self.seed))
                min_loss = loss
        print("Training complete.")
        train_logger.handlers.clear()

    def evaluate(self, mse_rauc, mse_ap, mse_f1):
        model = torch.load("model_seed{}.pth".format(self.seed))
        model.eval()
        mse_score, test_label = [], []
        for idx, x_input, onehot_y, y_label, group_label in tqdm(self.test_loader, delay=10):
            x_input = x_input.to(self.device)
            x_pred, z, masks = self.model(x_input)
            mse_batch = self.score_func(x_input, x_pred)
            mse_batch = mse_batch.data.cpu()
            mse_score.extend(mse_batch)
        label_array = self.test_loader.dataset.label_y
        group1_idx = self.test_loader.dataset.group1_idx
        group2_idx = self.test_loader.dataset.group2_idx
        if self.dataset == 'mnistandusps':
            top_k = [1200, 1000, 800]
        elif self.dataset == 'mnistinvert':
            top_k = [500, 400, 300]
        elif self.dataset == 'celebatab':
            top_k = [4500, 4000, 3500]
        elif self.dataset == 'compas':
            top_k = [350, 300, 250]
        else:
            top_k = [500, 400, 300]

        if self.ratio == 1:
            if self.dataset == 'mnistandusps':
                top_k = [650, 600, 550]
            elif self.dataset == 'compas':
                top_k = [80, 70, 60]
        elif self.ratio == 2:
            if self.dataset == 'mnistandusps':
                top_k = [1000, 900, 800]
            elif self.dataset == 'compas':
                top_k = [120, 110, 100]
        elif self.ratio == 3 and self.dataset == 'mnistandusps':
            top_k = [1200, 1000, 800]
        elif self.ratio == 4 and self.dataset == 'mnistandusps':
            top_k = [1200, 1000, 800]
        elif self.ratio == 5 and self.dataset == 'compas':
            top_k = [240, 220, 200]

        for k in top_k:
            print('When k=', k)
            # print(torch.tensor(mse_score).shape)
            idx = torch.topk(torch.tensor(mse_score), k=k)[1]
            pred = torch.zeros(label_array.shape[0]).to(self.device)
            pred[idx] = 1
            correct_pred = pred * label_array
            # Acc
            test_acc = accuracy(pred, label_array)
            group1_acc = accuracy(pred[group1_idx],
                                  label_array[group1_idx])
            group2_acc = accuracy(pred[group2_idx],
                                  label_array[group2_idx])
            # auc_roc = roc_auc_score(label_array[test_dataloader.dataset.group2_idx].cpu(), pred[test_dataloader.dataset.group2_idx].cpu())
            # auc_roc = roc_auc_score(label_array[test_dataloader.dataset.group2_idx].cpu(), mse_accumulate[test_dataloader.dataset.group2_idx].cpu())
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
            rocauc = roc_auc_score(label_array.cpu().numpy(), mse_score)
            acc_diff = abs(group1_acc - group2_acc).item()
            prec_diff = abs(precision_score(label_array[group1_idx].cpu(), pred[group1_idx].cpu())
                            - precision_score(label_array[group2_idx].cpu(), pred[group2_idx].cpu()))
            recall1 = recall_score(label_array[group1_idx].cpu(), pred[group1_idx].cpu())
            recall2 = recall_score(label_array[group2_idx].cpu(), pred[group2_idx].cpu())
            recall_diff = abs(recall1 - recall2)
            print(
                "Recall:  {:.4f}, Accuracy:  {:.4f}, ROCAUC:  {:.4f},  ACC Diff: {:.4f}, Precision Diff: {:.4f}, Recall Diff: {:.4f}".format(
                    recall, test_acc.item(), rocauc, acc_diff, prec_diff, recall_diff))


        # mse_rauc[self.run], mse_ap[self.run] = aucPerformance(mse_score, label_array)
        # mse_f1[self.run] = F1Performance(mse_score, label_array)