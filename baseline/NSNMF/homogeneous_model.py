import torch.nn as nn
import torch
import numpy as np
from sklearn.metrics import recall_score, roc_auc_score


class NSNMF(torch.nn.Module):
    def __init__(self, in_dim, k=10, n=1000, alpha=1):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n, k))
        self.H = nn.Parameter(torch.randn(k, in_dim))
        self.gamma = 1
        self.alpha = alpha
        self.n_cluster = k

    def forward(self, sim, feature):
        loss_1 = torch.norm(sim - torch.matmul(self.W, self.W.T), 2)
        loss_2 = torch.norm(feature - torch.matmul(self.W, self.H), 2) * self.alpha
        loss_3 = (torch.norm(self.W, 2) + torch.norm(self.H, 2)) * self.gamma
        loss = loss_3 + loss_2 + loss_1
        return loss

    def cosine_sim(self, x1, x2, temperature=1, eps=1e-15):
        w1 = x1.norm(p=2, dim=1, keepdim=True)
        w2 = x2.norm(p=2, dim=1, keepdim=True)
        return torch.matmul(x1, x2.t()) / ((w1 * w2.t()).clamp(min=eps) * temperature)

    def evaluation(self, data, feature, k):
        self.eval()
        loss = torch.norm(feature - torch.matmul(self.W, self.H), 2, dim=1)
        idx = torch.topk(loss, k=k)[1].cpu()
        y_pred1 = np.zeros(data.dataset.label_y.shape[0])
        y_pred1[idx] = 1
        recall1 = recall_score(data.dataset.label_y.cpu().numpy(), y_pred1)
        auc1 = roc_auc_score(data.dataset.label_y.cpu().numpy(), y_pred1)
        print('Recall:{:.4f},  AUC:{:.4f}'.format(recall1, auc1))
        return recall1, auc1

    def correct_pred(self, labels, y_pred):
        a = np.where(labels.cpu().numpy() == 1)[0].tolist()
        b = np.where(y_pred == 1)[0].tolist()
        a = set(a)
        b = set(b)
        return len(a.intersection(b))