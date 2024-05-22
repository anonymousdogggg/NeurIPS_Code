import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from utils import GumbelAcc, Samplewise_Weighted_CrossEntropyLoss, GumbelTNR, GumbelTPR


class Autoencoder(nn.Module):
    def __init__(self, args, input_d=32*32, layer=1, hid=128):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(input_d, hid), nn.ReLU(),)
        for i in range(layer - 1):
            self.encoder.append(nn.Linear(hid, hid))
            self.encoder.append(nn.ReLU())
        # self.encoder_projection = nn.Sequential(nn.Linear(2 * hid, hid), nn.ReLU())
        self.encoder_projection = nn.Sequential(nn.Linear(hid, hid), nn.ReLU())
        if layer == 1:
            self.decoder = nn.Sequential(nn.Linear(hid, input_d), nn.Sigmoid())
        else:
            self.decoder = nn.Sequential()
            for i in range(layer - 1):
                self.decoder.append(nn.Linear(hid, hid))
                self.decoder.append(nn.ReLU())
            self.decoder.append(nn.Linear(hid, input_d))
            if args.dataset in ['mnistandusps', 'mnistinvert', 'celebatab']:
                self.decoder.append(nn.Sigmoid())
            # else:
            #     self.decoder.append(nn.Sigmoid())
        # self.projection = nn.Sequential(nn.Linear(128, 50), nn.ReLU(inplace=True), nn.BatchNorm1d(50))
        self.proj_flag = args.projection
        self.projection = nn.Sequential(nn.Linear(hid, args.projection, bias=True)) # proj dimension= 50
        self.criterion = nn.CrossEntropyLoss()
        self.alpha = args.alpha
        self.gamma = args.gamma
        self.input_d = input_d

    def forward(self, x, group_label=None):
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)  # Flatten the input image
        h = self.encoder(x)
        if self.alpha != 0:
            contra_loss, old_weight = self.contrastive_loss(h, group_label)
        else:
            contra_loss = 0
            old_weight = 0
        x = self.decoder(h)
        if len(x.shape) > 2:
            x = x.view(x.size(0), self.input_d)
        return x, self.alpha * contra_loss, old_weight


    def reconstruction(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input image
        x = self.encoder(x)
        x = self.decoder(x)
        if len(x.shape) > 2:
            x = x.view(x.size(0), self.input_d)
        return x

    def cosine_sim(self, x1, x2, eps=1e-15, temperature=1):
        w1 = x1.norm(p=2, dim=1, keepdim=True)
        w2 = x2.norm(p=2, dim=1, keepdim=True)
        return torch.matmul(x1, x2.t()) / ((w1 * w2.t()).clamp(min=eps) * temperature)

    def get_embedding(self, x, group_label):
        x = self.encoder(x)
        z = self.projection(x)
        return z

    def contrastive_loss(self, x, group_label=None):
        if group_label is not None:
            group1_idx = torch.where(group_label == 0)[0]
            group2_idx = torch.where(group_label == 1)[0]
            if self.proj_flag > 0:
                z = self.projection(x)
            else:
                z = x
            group_1_z = z[group1_idx]
            group_2_z = z[group2_idx]
            if len(group_2_z) == 0 or len(group_1_z) == 0:
                contra_loss = 0
            else:
                pos_similarity = self.exp_cosine_sim(group_1_z, group_2_z).mean(dim=0)
                # pos_similarity = self.exp_cosine_sim(group_1_z, group_2_z).sum()
                # neg_similarity = (self.exp_cosine_sim(group_1_z, group_1_z).sum() +
                #                   self.exp_cosine_sim(group_2_z, group_2_z).sum())
                neg_similarity = (self.exp_cosine_sim(group_1_z, group_1_z).mean() + self.gamma *
                                  self.exp_cosine_sim(group_2_z, group_2_z).mean())
                contra_loss = -1 * torch.mean(torch.log(pos_similarity/(neg_similarity)))
                if torch.isnan(contra_loss):
                    print('error detected, nan')
                weight = (self.exp_cosine_sim(group_1_z, group_1_z).mean() /
                      (self.exp_cosine_sim(group_1_z, group_1_z).mean() + self.gamma * self.exp_cosine_sim(group_2_z,
                                                                                                           group_2_z).mean()))

        else:
            contra_loss = 0
            weight = 0
        return contra_loss, weight

    def exp_cosine_sim(self, x1, x2, eps=1e-15, temperature=1):
        w1 = x1.norm(p=2, dim=1, keepdim=True)
        w2 = x2.norm(p=2, dim=1, keepdim=True)
        return torch.exp(torch.matmul(x1, x2.t()) / ((w1 * w2.t()).clamp(min=eps) * temperature))

    def contrastive_loss_entropy_form(self, x, group_label=None):
        if group_label != None:
            group1_idx = torch.where(group_label == 0)[0]
            group2_idx = torch.where(group_label == 1)[0]
            z = self.projection(x)
            group_1_z = z[group1_idx]
            group_2_z = z[group2_idx]
            if len(group_2_z) == 0 or len(group_1_z) == 0:
                contra_loss = 0
            else:
                pos_similarity = self.exp_cosine_sim(group_1_z, group_2_z).reshape(-1, 1)
                neg_similarity = torch.cat((self.exp_cosine_sim(group_1_z, group_1_z).reshape(1, -1),
                                            self.exp_cosine_sim(group_2_z, group_2_z).reshape(1, -1)), 1)
                N = pos_similarity.shape[0]
                labels = torch.zeros(N).to(pos_similarity.device).long()
                logits = torch.cat((pos_similarity, neg_similarity.expand(N, neg_similarity.shape[1])), dim=1)
                contra_loss = self.criterion(logits, labels)
                if torch.isnan(contra_loss):
                    print('error detected, nan')
        else:
            contra_loss = 0
        return contra_loss


class MLP(torch.nn.Module):
    def __init__(self, args):
        super(MLP, self).__init__()
        input_size, hidden_size, output_dim = args.raw_dim, args.hidden_dim, args.num_class
        self.args = args
        self.input_size = input_size
        self.hidden_size = hidden_size
        if args.model == 'logistic':
            self.fc1 = torch.nn.Linear(self.input_size, output_dim, bias=False)
        else:
            self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
            self.relu = torch.nn.ReLU()
            self.fc2 = torch.nn.Linear(self.hidden_size, output_dim)
        self.approxi_acc = GumbelAcc()
        self.approxi_tpr = GumbelTPR()
        self.approxi_tnr = GumbelTNR()
        self.ce_loss = Samplewise_Weighted_CrossEntropyLoss()
        self.ce_loss_sum = Samplewise_Weighted_CrossEntropyLoss(reduction='sum')

    def forward(self, x):
        if self.args.norm_input:
            x = F.normalize(x, p=2, dim=-1)

        if self.args.model == 'logistic':
            output = self.fc1(x.reshape(x.shape[0], -1))
        else:
            hidden = self.fc1(x.reshape(x.shape[0], -1))
            hidden = self.relu(hidden)
            output = self.fc2(hidden)
        return output


    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


    def count_parameters_each_layer(self):
        return [p.numel() for p in self.parameters() if p.requires_grad]


    def collect_grad(self):
        return torch.cat([p.grad.reshape(-1,) for p in self.parameters() if p.requires_grad], dim=0)


    def collect_batch_grad(self, params=None):
        batch_grad_cache = []
        if params is not None:
            for param in params:
                if param.requires_grad:
                    batch_grad_cache.append(param.grad_batch.reshape(param.grad_batch.shape[0], -1))
        else:
            for name, param in self.named_parameters():
                if param.requires_grad:
                    batch_grad_cache.append(param.grad_batch.reshape(param.grad_batch.shape[0], -1))

        batch_grad_cache = torch.cat(batch_grad_cache, dim=1)
        return batch_grad_cache


    def separate_batch_grad(self, batch_grad_cache, params=None):
        num_param_per_layer = []
        if params is not None:
            for param in params:
                temp_num_param = np.prod(list(param.shape))
                num_param_per_layer.append(temp_num_param)
        else:
            for name, param in self.named_parameters():
                temp_num_param = np.prod(list(param.shape))
                num_param_per_layer.append(temp_num_param)

        grad_per_layer_list = []
        counter = 0
        for num_param in num_param_per_layer:
            grad_per_layer_list.append(batch_grad_cache[counter:counter+num_param])
            counter += num_param

        return grad_per_layer_list



class LeNet(nn.Module):
    def __init__(self, args):
        super(LeNet, self).__init__()
        input_size, hidden_size, output_dim = args.raw_dim, args.hidden_dim, args.num_class
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6,
                               kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16,
                               kernel_size=5, stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120,
                               kernel_size=5, stride=1, padding=0)
        self.linear1 = nn.Linear(120, 84)
        self.linear2 = nn.Linear(84, output_dim)
        self.fc_in_dim = 84
        self.tanh = nn.Tanh()
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.approxi_acc = GumbelAcc()
        self.approxi_tpr = GumbelTPR()
        self.approxi_tnr = GumbelTNR()
        self.ce_loss = Samplewise_Weighted_CrossEntropyLoss()
        self.ce_loss_sum = Samplewise_Weighted_CrossEntropyLoss(reduction='sum')

    def forward(self, x):
        x = self.conv1(x)
        x = self.tanh(x)
        x = self.avgpool(x)
        x = self.conv2(x)
        x = self.tanh(x)
        x = self.avgpool(x)
        x = self.conv3(x)
        x = self.tanh(x)

        x = x.reshape(x.shape[0], -1)
        x = self.linear1(x)
        x = self.tanh(x)
        x = self.linear2(x)
        return x

    def cfair_forward(self, x):
        x = self.conv1(x)
        x = self.tanh(x)
        x = self.avgpool(x)
        x = self.conv2(x)
        x = self.tanh(x)
        x = self.avgpool(x)
        x = self.conv3(x)
        x = self.tanh(x)

        x = x.reshape(x.shape[0], -1)
        x = self.linear1(x)
        x = self.tanh(x)
        return x


    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


    def collect_batch_grad(self, params=None):
        batch_grad_cache = []
        if params is not None:
            for param in params:
                if param.requires_grad:
                    batch_grad_cache.append(param.grad_batch.reshape(param.grad_batch.shape[0], -1))
        else:
            for name, param in self.named_parameters():
                if param.requires_grad:
                    batch_grad_cache.append(param.grad_batch.reshape(param.grad_batch.shape[0], -1))

        batch_grad_cache = torch.cat(batch_grad_cache, dim=1)
        return batch_grad_cache

    def separate_batch_grad(self, batch_grad_cache, params=None):
        num_param_per_layer = []
        if params is not None:
            for param in params:
                temp_num_param = np.prod(list(param.shape))
                num_param_per_layer.append(temp_num_param)
        else:
            for name, param in self.named_parameters():
                temp_num_param = np.prod(list(param.shape))
                num_param_per_layer.append(temp_num_param)

        grad_per_layer_list = []
        counter=0
        for num_param in num_param_per_layer:
            grad_per_layer_list.append(batch_grad_cache[counter:counter+num_param])
            counter += num_param

        return grad_per_layer_list



class CNN(nn.Module):
    def __init__(self, args):
        super().__init__()
        output_dim = args.num_class
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, output_dim)
        self.fc_in_dim = 16 * 5 * 5
        self.approxi_acc = GumbelAcc()
        self.approxi_tpr = GumbelTPR()
        self.approxi_tnr = GumbelTNR()
        self.ce_loss = Samplewise_Weighted_CrossEntropyLoss()
        self.ce_loss_sum = Samplewise_Weighted_CrossEntropyLoss(reduction='sum')


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.fc1(x)
        return x

    def cfair_forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        return x


    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


    def collect_batch_grad(self, params=None):
        batch_grad_cache = []
        if params is not None:
            for param in params:
                if param.requires_grad:
                    batch_grad_cache.append(param.grad_batch.reshape(param.grad_batch.shape[0], -1))
        else:
            for name, param in self.named_parameters():
                if param.requires_grad:
                    batch_grad_cache.append(param.grad_batch.reshape(param.grad_batch.shape[0], -1))

        batch_grad_cache = torch.cat(batch_grad_cache, dim=1)
        return batch_grad_cache

    def separate_batch_grad(self, batch_grad_cache, params=None):
        num_param_per_layer = []
        if params is not None:
            for param in params:
                temp_num_param = np.prod(list(param.shape))
                num_param_per_layer.append(temp_num_param)
        else:
            for name, param in self.named_parameters():
                temp_num_param = np.prod(list(param.shape))
                num_param_per_layer.append(temp_num_param)

        grad_per_layer_list = []
        counter=0
        for num_param in num_param_per_layer:
            grad_per_layer_list.append(batch_grad_cache[counter:counter+num_param])
            counter += num_param

        return grad_per_layer_list



cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512,
              'M'],
}



class VGG(nn.Module):
    def __init__(self, vgg_name='VGG11'):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)
        self.approxi_acc = GumbelAcc()
        self.ce_loss = Samplewise_Weighted_CrossEntropyLoss()
        self.ce_loss_sum = Samplewise_Weighted_CrossEntropyLoss(reduction='sum')


    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU()]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


    def collect_batch_grad(self, params=None):
        batch_grad_cache = []
        if params is not None:
            for param in params:
                if param.requires_grad:
                    batch_grad_cache.append(param.grad_batch.reshape(param.grad_batch.shape[0], -1))
        else:
            for name, param in self.named_parameters():
                if param.requires_grad:
                    batch_grad_cache.append(param.grad_batch.reshape(param.grad_batch.shape[0], -1))

        batch_grad_cache = torch.cat(batch_grad_cache, dim=1)
        return batch_grad_cache


    def separate_batch_grad(self, batch_grad_cache, params=None):
        num_param_per_layer = []
        if params is not None:
            for param in params:
                temp_num_param = np.prod(list(param.shape))
                num_param_per_layer.append(temp_num_param)
        else:
            for name, param in self.named_parameters():
                temp_num_param = np.prod(list(param.shape))
                num_param_per_layer.append(temp_num_param)

        grad_per_layer_list = []
        counter=0
        for num_param in num_param_per_layer:
            grad_per_layer_list.append(batch_grad_cache[counter:counter+num_param])
            counter += num_param

        return grad_per_layer_list

