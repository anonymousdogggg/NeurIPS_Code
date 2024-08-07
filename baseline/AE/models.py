import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from utils import GumbelAcc, Samplewise_Weighted_CrossEntropyLoss, GumbelTNR, GumbelTPR


class Autoencoder(nn.Module):
    def __init__(self, args):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(32 * 32, 128),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(128, 32 * 32),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input image
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.view(x.size(0), 1, 32 * 32)
        return x


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

