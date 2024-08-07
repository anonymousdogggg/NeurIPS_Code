import torch
import numpy as np
import pickle as pkl
import os
import torch.nn.functional as F
import scipy.io as sio


class Graph(object):
    n_feats = None
    n_class = None
    n_nodes = None
    feats = None
    labels = None
    train_idx = None
    test_idx = None


def load_dblp_graph():
    graph = Graph()
    # {'adj_matrix': graph, 'label': label, 'node_id': node_dict, 'feature': node_feature}
    if not os.path.exists('../../data/DBLP_anomaly.mat'):
        data = sio.loadmat('../../data/DBLP')
        feature = data['features'].astype(np.float64)
        label = np.zeros((feature.shape[0], 1))
        p = 0.07
        index = np.random.choice(feature.shape[0], int(p * feature.shape[0]), replace=False)
        label[index] = 1
        feature_anomaly_idx = index[:index.shape[0]//2]
        edge_anomaly_idx = index[index.shape[0]//2:]
        feature[feature_anomaly_idx] += np.random.normal(2, 10, size=(feature_anomaly_idx.shape[0], feature.shape[1]))
        # graph.adj1 = np.dot(data['net_APA'], data['net_APA'].transpose())
        # graph.adj2 = np.dot(data['net_APCPA'], data['net_APCPA'].transpose())
        graph.adj1 = data['net_APA']
        graph.adj2 = data['net_APCPA']
        graph.v1 = feature
        graph.v2 = feature
        graph.adj1[edge_anomaly_idx, :] += np.random.binomial(1, 0.5, size=(edge_anomaly_idx.shape[0], graph.adj1.shape[1]))
        graph.adj2[edge_anomaly_idx, :] += np.random.binomial(1, 0.5, size=(edge_anomaly_idx.shape[0], graph.adj2.shape[1]))
        graph.adj1 = (graph.adj1 > 0) + 0
        graph.adj2 = (graph.adj2 > 0) + 0
        data = {'feature': feature, 'label': label, 'adj1': graph.adj1, 'adj2': graph.adj2}
        sio.savemat('../../data/dblp_anomaly.mat', data)
    else:
        data = sio.loadmat('../../data/dblp_anomaly.mat')
        graph.v1 = data['feature']
        graph.v2 = data['feature']
        graph.adj1 = data['adj1']
        graph.adj2 = data['adj2']
        label = data['label']
    graph.v1 = torch.FloatTensor(graph.v1)
    graph.v2 = torch.FloatTensor(graph.v2)
    # graph.v1 = F.normalize(graph.v1, p=2, dim=1)
    # graph.v2 = F.normalize(graph.v2, p=2, dim=1)
    graph.adj1 = torch.FloatTensor(graph.adj1)
    graph.adj2 = torch.FloatTensor(graph.adj2)
    graph.labels = torch.LongTensor(np.array(label))
    graph.n_nodes = graph.v1.shape[0]
    graph.n_feats = graph.v1.shape[1]
    graph.n_classes = 3
    # print('size of anomalies:', sum(label))
    return graph

def load_cert_graph(d):
    graph = Graph()
    v2 = pkl.load(open('../../CERT/logon.pkl', 'rb'))
    v1 = pkl.load(open('../../CERT/email.pkl', 'rb'))
    malicious_user = pkl.load(open('../../CERT/label.pkl', 'rb'))['label']
    label = []
    email_pc_dict = v2['pc_dict']
    email_user_dict = v2['user_dict']
    overlapped_idx = []
    for item, key in email_pc_dict.items():
        if item in v1['pc_dict']:
            overlapped_idx.append(v1['pc_dict'][item])
    v2['graph'] = v2['graph'][overlapped_idx, :]
    v2['weight'] = v2['weight'][overlapped_idx, :]
    overlapped_idx = []
    for item, key in email_user_dict.items():
        if item in v1['user_dict']:
            overlapped_idx.append(v1['user_dict'][item])
        if item in malicious_user:
            label.append(1)
        else:
            label.append(0)
    v2['graph'] = v2['graph'][:, overlapped_idx]
    v2['weight'] = v2['weight'][:, overlapped_idx]
    if not os.path.exists('../../CERT/logon_edge_list.txt'):
        n_nodes = v2['weight'].shape[0]
        with open('../../CERT/logon_edge_list.txt', 'w') as f:
            for i in range(len(v2['graph'])):
                idx = np.nonzero(v2['graph'][i, :])[0]
                for j in idx:
                    f.write('{} {}\n'.format(i, j+n_nodes))
    if not os.path.exists('../../CERT/email_edge_list.txt'):
        n_nodes = v1['weight'].shape[0]
        with open('../../CERT/email_edge_list.txt', 'w') as f:
            for i in range(len(v1['graph'])):
                idx = np.nonzero(v1['graph'][i, :])[0]
                for j in idx:
                    f.write('{} {}\n'.format(i, j+n_nodes))
    if not os.path.exists('../../CERT/email_edge_list_emb_{}'.format(d)):
        os.system("python ../../deepwalk/main.py --representation-size {} --input ../../CERT/email_edge_list.txt"
                  " --output ../../CERT/email_edge_list_emb_{}".format(d, d))
    if not os.path.exists('../../CERT/logon_edge_list_emb_{}'.format(d)):
        os.system("python ../../deepwalk/main.py --representation-size {} --input ../../CERT/logon_edge_list.txt"
                  " --output ../../CERT/logon_edge_list_emb_{}".format(d, d))
    graph.u1 = np.zeros((v1['weight'].shape[0], d))
    graph.v1 = np.zeros((v1['weight'].shape[1], d))
    graph.u2 = np.zeros((v2['weight'].shape[0], d))
    graph.v2 = np.zeros((v2['weight'].shape[1], d))
    with open('../../CERT/logon_edge_list_emb_{}'.format(d), 'r') as f:
        next(f)
        for line in f.readlines():
            line = list(map(float, line.split()))
            if line[0] >= 1000:
                graph.v2[int(line[0])-1000] = line[1:]
            else:
                graph.u2[int(line[0])] = line[1:]
    with open('../../CERT/email_edge_list_emb_{}'.format(d), 'r') as f:
        next(f)
        for line in f.readlines():
            line = list(map(float, line.split()))
            if line[0] >= 1000:
                graph.v1[int(line[0]) - 1000] = line[1:]
            else:
                graph.u1[int(line[0])] = line[1:]
    # graph.u1 = np.ones((v1['weight'].shape[0], d))
    # graph.v1 = np.ones((v1['weight'].shape[1], d))
    # graph.u2 = np.ones((v2['weight'].shape[0], d))
    # graph.v2 = np.ones((v2['weight'].shape[1], d))
    graph.train_idx = torch.arange(0, 100)
    graph.test_idx = torch.arange(100, 1000)
    graph.u1 = torch.FloatTensor(graph.u1)
    graph.v1 = torch.FloatTensor(graph.v1)
    graph.u2 = torch.FloatTensor(graph.u2)
    graph.v2 = torch.FloatTensor(graph.v2)
    graph.u1 = graph.u1 / torch.norm(graph.u1, p=2, dim=1, keepdim=True)
    graph.u2 = graph.u2 / torch.norm(graph.u2, p=2, dim=1, keepdim=True)
    graph.v1 = graph.v1 / torch.norm(graph.v1, p=2, dim=1, keepdim=True)
    graph.v2 = graph.v2 / torch.norm(graph.v2, p=2, dim=1, keepdim=True)
    nan_idx = torch.isnan(graph.v2)
    graph.v2[nan_idx] = 0
    v1_feats = v1['weight']
    v2_feats = v2['weight']
    v1_adj = v1['graph']
    v2_adj = v2['graph']
    graph.adj1 = torch.FloatTensor(v1_adj)
    graph.adj2 = torch.FloatTensor(v2_adj)
    graph.w1 = torch.FloatTensor(v1_feats)
    graph.w1 = graph.w1 / graph.w1.sum(dim=1).reshape(-1, 1)
    graph.w2 = torch.FloatTensor(v2_feats)
    graph.w2 = graph.w1 / graph.w2.sum(dim=1).reshape(-1, 1)
    graph.labels = torch.LongTensor(np.array(label)).reshape(-1, 1)
    graph.n_feats = graph.u1.shape[1]
    graph.n_classes = 2
    graph.n_nodes = v1_feats.shape[0]
    return graph


#
# def load_cert_graph(d):
#     graph = Graph()
#     v2 = pkl.load(open('./CERT/r4.2/logon.pkl', 'rb'))
#     v1 = pkl.load(open('./CERT/r4.2/email.pkl', 'rb'))
#     malicious_user = pkl.load(open('./CERT/r4.2/label.pkl', 'rb'))['label']
#     label = []
#     email_pc_dict = v2['pc_dict']
#     email_user_dict = v2['user_dict']
#     overlapped_idx = []
#     for item, key in email_pc_dict.items():
#         if item in v1['pc_dict']:
#             overlapped_idx.append(v1['pc_dict'][item])
#     v2['graph'] = v2['graph'][overlapped_idx, :]
#     v2['weight'] = v2['weight'][overlapped_idx, :]
#     overlapped_idx = []
#     for item, key in email_user_dict.items():
#         if item in v1['user_dict']:
#             overlapped_idx.append(v1['user_dict'][item])
#         if item in malicious_user:
#             label.append(1)
#         else:
#             label.append(0)
#     v2['graph'] = v2['graph'][:, overlapped_idx]
#     v2['weight'] = v2['weight'][:, overlapped_idx]
#     if not os.path.exists('./CERT/r4.2/logon_edge_list.txt'):
#         n_nodes = v2['weight'].shape[0]
#         with open('./CERT/r4.2/logon_edge_list.txt', 'w') as f:
#             for i in range(len(v2['graph'])):
#                 idx = np.nonzero(v2['graph'][i, :])[0]
#                 for j in idx:
#                     f.write('{} {}\n'.format(i, j+n_nodes))
#     if not os.path.exists('./CERT/r4.2/email_edge_list.txt'):
#         n_nodes = v1['weight'].shape[0]
#         with open('./CERT/r4.2/email_edge_list.txt', 'w') as f:
#             for i in range(len(v1['graph'])):
#                 idx = np.nonzero(v1['graph'][i, :])[0]
#                 for j in idx:
#                     f.write('{} {}\n'.format(i, j+n_nodes))
#     if not os.path.exists('./CERT/r4.2/email_edge_list_emb_{}'.format(d)):
#         os.system("python ./deepwalk/main.py --representation-size {} --input ./CERT/r4.2/email_edge_list.txt"
#                   " --output ./CERT/r4.2/email_edge_list_emb_{}".format(d, d))
#     if not os.path.exists('./CERT/r4.2/logon_edge_list_emb_{}'.format(d)):
#         os.system("python ./deepwalk/main.py --representation-size {} --input ./CERT/r4.2/logon_edge_list.txt"
#                   " --output ./CERT/r4.2/logon_edge_list_emb_{}".format(d, d))
#     graph.u1 = np.zeros((v1['weight'].shape[0], d))
#     graph.v1 = np.zeros((v1['weight'].shape[1], d))
#     graph.u2 = np.zeros((v2['weight'].shape[0], d))
#     graph.v2 = np.zeros((v2['weight'].shape[1], d))
#     with open('./CERT/r4.2/logon_edge_list_emb_{}'.format(d), 'r') as f:
#         next(f)
#         for line in f.readlines():
#             line = list(map(float, line.split()))
#             if line[0] >= 1000:
#                 graph.v2[int(line[0])-1000] = line[1:]
#             else:
#                 graph.u2[int(line[0])] = line[1:]
#     with open('./CERT/r4.2/email_edge_list_emb_{}'.format(d), 'r') as f:
#         next(f)
#         for line in f.readlines():
#             line = list(map(float, line.split()))
#             if line[0] >= 1000:
#                 graph.v1[int(line[0]) - 1000] = line[1:]
#             else:
#                 graph.u1[int(line[0])] = line[1:]
#     graph.train_idx = torch.arange(0, 100)
#     graph.test_idx = torch.arange(100, 1000)
#     graph.u1 = torch.FloatTensor(graph.u1)
#     graph.v1 = torch.FloatTensor(graph.v1)
#     graph.u2 = torch.FloatTensor(graph.u2)
#     graph.v2 = torch.FloatTensor(graph.v2)
#     graph.u1 = graph.u1 / torch.norm(graph.u1, p=2, dim=1, keepdim=True)
#     graph.u2 = graph.u2 / torch.norm(graph.u2, p=2, dim=1, keepdim=True)
#     graph.v1 = graph.v1 / torch.norm(graph.v1, p=2, dim=1, keepdim=True)
#     graph.v2 = graph.v2 / torch.norm(graph.v2, p=2, dim=1, keepdim=True)
#     nan_idx = torch.isnan(graph.v2)
#     graph.v2[nan_idx] = 0
#     v1_feats = v1['weight']
#     v2_feats = v2['weight']
#     v1_adj = v2['graph']
#     adj = torch.FloatTensor(np.dot(v1_adj, v1_adj.T))
#     adj = adj / adj.sum(dim=1, keepdim=True)
#     graph.adj1 = adj
#     graph.adj2 = adj
#     graph.w1 = torch.FloatTensor(v1_feats)
#     # graph.w1 = graph.w1 / graph.w1.sum(dim=1).reshape(-1, 1)
#     graph.w2 = torch.FloatTensor(v2_feats)
#     # graph.w2 = graph.w1 / graph.w2.sum(dim=1).reshape(-1, 1)
#     graph.labels = torch.LongTensor(np.array(label))
#     graph.n_feats = graph.u1.shape[1]
#     graph.n_classes = 2
#     graph.n_nodes = v1_feats.shape[0]
#     return graph



#
# def load_cert_graph(d):
#     graph = Graph()
#     data = pkl.load(open('./cert.pkl', 'rb'))
#     # v2 = pkl.load(open('./CERT/r4.2/logon.pkl', 'rb'))
#     # v1 = pkl.load(open('./CERT/r4.2/email.pkl', 'rb'))
#     # malicious_user = pkl.load(open('./CERT/r4.2/label.pkl', 'rb'))['label']
#     # label = []
#     # email_pc_dict = v2['pc_dict']
#     # email_user_dict = v2['user_dict']
#     # overlapped_idx = []
#     # for item, key in email_pc_dict.items():
#     #     if item in v1['pc_dict']:
#     #         overlapped_idx.append(v1['pc_dict'][item])
#     # v2['graph'] = v2['graph'][overlapped_idx, :]
#     # v2['weight'] = v2['weight'][overlapped_idx, :]
#     # overlapped_idx = []
#     # for item, key in email_user_dict.items():
#     #     if item in v1['user_dict']:
#     #         overlapped_idx.append(v1['user_dict'][item])
#     #     if item in malicious_user:
#     #         label.append(1)
#     #     else:
#     #         label.append(0)
#     # v2['graph'] = v2['graph'][:, overlapped_idx]
#     # v2['weight'] = v2['weight'][:, overlapped_idx]
#     # if not os.path.exists('./CERT/r4.2/logon_edge_list.txt'):
#     #     n_nodes = v2['weight'].shape[0]
#     #     with open('./CERT/r4.2/logon_edge_list.txt', 'w') as f:
#     #         for i in range(len(v2['graph'])):
#     #             idx = np.nonzero(v2['graph'][i, :])[0]
#     #             for j in idx:
#     #                 f.write('{} {}\n'.format(i, j+n_nodes))
#     # if not os.path.exists('./CERT/r4.2/email_edge_list.txt'):
#     #     n_nodes = v1['weight'].shape[0]
#     #     with open('./CERT/r4.2/email_edge_list.txt', 'w') as f:
#     #         for i in range(len(v1['graph'])):
#     #             idx = np.nonzero(v1['graph'][i, :])[0]
#     #             for j in idx:
#     #                 f.write('{} {}\n'.format(i, j+n_nodes))
#     # if not os.path.exists('./CERT/r4.2/email_edge_list_emb_{}'.format(d)):
#     #     os.system("python ./deepwalk/main.py --representation-size {} --input ./CERT/r4.2/email_edge_list.txt"
#     #               " --output ./CERT/r4.2/email_edge_list_emb_{}".format(d, d))
#     # if not os.path.exists('./CERT/r4.2/logon_edge_list_emb_{}'.format(d)):
#     #     os.system("python ./deepwalk/main.py --representation-size {} --input ./CERT/r4.2/logon_edge_list.txt"
#     #               " --output ./CERT/r4.2/logon_edge_list_emb_{}".format(d, d))
#     # graph.u1 = np.zeros((v1['weight'].shape[0], d))
#     # graph.v1 = np.zeros((v1['weight'].shape[1], d))
#     # graph.u2 = np.zeros((v2['weight'].shape[0], d))
#     # graph.v2 = np.zeros((v2['weight'].shape[1], d))
#     # with open('./CERT/r4.2/logon_edge_list_emb_{}'.format(d), 'r') as f:
#     #     next(f)
#     #     for line in f.readlines():
#     #         line = list(map(float, line.split()))
#     #         if line[0] >= 1000:
#     #             graph.v2[int(line[0])-1000] = line[1:]
#     #         else:
#     #             graph.u2[int(line[0])] = line[1:]
#     # with open('./CERT/r4.2/email_edge_list_emb_{}'.format(d), 'r') as f:
#     #     next(f)
#     #     for line in f.readlines():
#     #         line = list(map(float, line.split()))
#     #         if line[0] >= 1000:
#     #             graph.v1[int(line[0]) - 1000] = line[1:]
#     #         else:
#     #             graph.u1[int(line[0])] = line[1:]
#     # graph.train_idx = torch.arange(0, 100)
#     # graph.test_idx = torch.arange(100, 1000)
#     # graph.u1 = torch.FloatTensor(graph.u1)
#     # graph.v1 = torch.FloatTensor(graph.v1)
#     # graph.u2 = torch.FloatTensor(graph.u2)
#     # graph.v2 = torch.FloatTensor(graph.v2)
#     # graph.u1 = graph.u1 / torch.norm(graph.u1, p=2, dim=1, keepdim=True)
#     # graph.u2 = graph.u2 / torch.norm(graph.u2, p=2, dim=1, keepdim=True)
#     # graph.v1 = graph.v1 / torch.norm(graph.v1, p=2, dim=1, keepdim=True)
#     # graph.v2 = graph.v2 / torch.norm(graph.v2, p=2, dim=1, keepdim=True)
#     # nan_idx = torch.isnan(graph.v2)
#     # graph.v2[nan_idx] = 0
#     # v1_feats = v1['weight']
#     # v2_feats = v2['weight']
#     # v1_adj = v2['graph']
#     # adj = torch.FloatTensor(np.dot(v1_adj, v1_adj.T))
#     # adj = adj / adj.sum(dim=1, keepdim=True)
#     graph.adj1 = torch.FloatTensor(data['adj'])
#     graph.adj2 = torch.FloatTensor(data['adj'])
#     graph.v1 = data['feat_v1']
#     graph.v2 = data['feat_v2']
#     graph.labels = torch.LongTensor(np.array(data['label']))
#     graph.n_feats = graph.v1.shape[1]
#     graph.n_classes = 2
#     graph.n_nodes = graph.v1.shape[0]
#     return graph


def load_cmce_graph(d):
    graph = Graph()
    # {'adj_matrix': graph, 'label': label, 'node_id': node_dict, 'feature': node_feature}
    data = pkl.load(open('./data/authentication_flow_small_500000.pkl', 'rb'))
    label = data['label']
    feature = data['feature']
    graph.adj1 = data['adj_matrix']
    graph.adj2 = data['adj_matrix']
    graph.v1 = feature[:, :200]
    graph.v2 = feature[:, 200:]
    graph.train_idx = torch.arange(0, 100)
    graph.test_idx = torch.arange(100, 1000)
    graph.v1 = torch.FloatTensor(graph.v1)
    graph.v2 = torch.FloatTensor(graph.v2)
    graph.v1 = F.normalize(graph.v1, p=2, dim=1)
    graph.v2 = F.normalize(graph.v2, p=2, dim=1)
    graph.adj1 = torch.FloatTensor(graph.adj1)
    graph.adj2 = torch.FloatTensor(graph.adj2)
    # nan_idx = graph.v2.isnan()
    # graph.v2[nan_idx] = 0
    graph.labels = torch.LongTensor(np.array(label))
    graph.n_nodes = graph.v1.shape[0]
    graph.n_feats = graph.v1.shape[1]
    graph.n_classes = 2
    return graph


def sigmoid(data):
    return 1/(1+np.exp(-data))


# def tanh(data):
#     return np.tanh(data)


def load_blog_graph():
    graph = Graph()
    data = sio.loadmat('../../data/BLOG.mat')
    feature = data['Attributes'].todense()
    graph.adj1 = data['Network'].todense() + np.identity(data['Network'].shape[0])
    graph.adj2 = data['Network'].todense() + np.identity(data['Network'].shape[0])
    graph.v1 = sigmoid(feature)
    graph.v2 = feature
    graph.train_idx = torch.arange(0, 100)
    graph.test_idx = torch.arange(100, 1000)
    graph.v1 = torch.FloatTensor(graph.v1)
    graph.v2 = torch.FloatTensor(graph.v2)
    # graph.v1 = F.normalize(graph.v1, p=2, dim=1)
    # graph.v2 = F.normalize(graph.v2, p=2, dim=1)
    graph.adj1 = torch.FloatTensor(graph.adj1)
    graph.adj2 = torch.FloatTensor(graph.adj2)
    graph.adj1 = graph.adj1 / graph.adj1.sum(dim=1, keepdim=True)
    graph.adj2 = graph.adj2 / graph.adj2.sum(dim=1, keepdim=True)
    # nan_idx = graph.v2.isnan()
    # graph.v2[nan_idx] = 0
    graph.labels = torch.LongTensor(np.array(data['Label']))
    graph.n_nodes = graph.v1.shape[0]
    graph.n_feats = graph.v1.shape[1]
    graph.n_classes = 2
    return graph


#
# def load_acm_graph(d):
#     graph = Graph()
#     # {'adj_matrix': graph, 'label': label, 'node_id': node_dict, 'feature': node_feature}
#     # data = pkl.load(open('./data/authentication_flow_small_500000.pkl', 'rb'))
#     data = sio.loadmat('./data/ACM.mat')
#     feature = data['Attributes'].todense()
#     graph.adj1 = data['Network'].todense()
#     graph.adj2 = data['Network'].todense()
#     graph.v1 = feature[:, :4000]
#     graph.v2 = feature[:, 4000:8000]
#     graph.train_idx = torch.arange(0, 100)
#     graph.test_idx = torch.arange(100, 1000)
#     graph.v1 = torch.FloatTensor(graph.v1)
#     graph.v2 = torch.FloatTensor(graph.v2)
#     graph.v1 = F.normalize(graph.v1, p=2, dim=1)
#     graph.v2 = F.normalize(graph.v2, p=2, dim=1)
#     graph.adj1 = torch.FloatTensor(graph.adj1)
#     graph.adj2 = torch.FloatTensor(graph.adj2)
#     # nan_idx = graph.v2.isnan()
#     # graph.v2[nan_idx] = 0
#     graph.labels = torch.LongTensor(np.array(data['Label']))
#     graph.n_nodes = graph.v1.shape[0]
#     graph.n_feats = graph.v1.shape[1]
#     graph.n_classes = 2
#     return graph


def load_imdb_graph():
    graph = Graph()
    # {'adj_matrix': graph, 'label': label, 'node_id': node_dict, 'feature': node_feature}
    if not os.path.exists('../../data/imdb5k_anomaly.mat'):
        data = sio.loadmat('../../data/imdb5k')
        feature = data['feature'].astype(np.float64)
        label = np.zeros((feature.shape[0], 1))
        p = 0.07
        index = np.random.choice(feature.shape[0], int(p * feature.shape[0]), replace=False)
        label[index] = 1
        feature_anomaly_idx = index[:index.shape[0]//2]
        edge_anomaly_idx = index[index.shape[0]//2:]
        feature[feature_anomaly_idx] += np.random.normal(0, 1, size=(feature_anomaly_idx.shape[0], feature.shape[1]))
        graph.adj1 = data['MAM']
        graph.adj2 = data['MDM']
        graph.v1 = feature
        graph.v2 = feature
        graph.adj1[edge_anomaly_idx, :] += np.random.binomial(1, 0.5, size=(edge_anomaly_idx.shape[0], graph.adj1.shape[1]))
        graph.adj2[edge_anomaly_idx, :] += np.random.binomial(1, 0.5, size=(edge_anomaly_idx.shape[0], graph.adj2.shape[1]))
        graph.adj1 = (graph.adj1 > 0) + 0
        graph.adj2 = (graph.adj2 > 0) + 0
        data = {'feature': feature, 'label': label, 'adj1': graph.adj1, 'adj2': graph.adj2}
        sio.savemat('../../data/imdb5k_anomaly.mat', data)
    else:
        data = sio.loadmat('../../data/imdb5k_anomaly.mat')
        graph.v1 = data['feature']
        graph.v2 = data['feature']
        graph.adj1 = data['adj1']
        graph.adj2 = data['adj2']
        label = data['label']
    graph.v1 = torch.FloatTensor(graph.v1)
    graph.v2 = torch.FloatTensor(graph.v2)
    graph.v1 = F.normalize(graph.v1, p=2, dim=1)
    graph.v2 = F.normalize(graph.v2, p=2, dim=1)
    graph.adj1 = torch.FloatTensor(graph.adj1)
    graph.adj2 = torch.FloatTensor(graph.adj2)
    graph.labels = torch.LongTensor(np.array(label))
    graph.n_nodes = graph.v1.shape[0]
    graph.n_feats = graph.v1.shape[1]
    graph.n_classes = 3
    print('size of anomalies:', sum(label))
    return graph

def load_acm_graph():
    graph = Graph()
    data = sio.loadmat('../../data/ACM.mat')
    feature = data['Attributes'].todense()
    graph.adj1 = data['Network'].todense() + np.identity(data['Network'].shape[0])
    graph.adj2 = data['Network'].todense() + np.identity(data['Network'].shape[0])
    graph.v1 = sigmoid(feature)
    # graph.v2 = np.tanh(feature)
    graph.v2 = feature
    graph.train_idx = torch.arange(0, 100)
    graph.test_idx = torch.arange(100, 1000)
    graph.v1 = torch.FloatTensor(graph.v1)
    graph.v2 = torch.FloatTensor(graph.v2)
    # graph.v1 = F.normalize(graph.v1, p=2, dim=1)
    # graph.v2 = F.normalize(graph.v2, p=2, dim=1)
    graph.adj1 = torch.FloatTensor(graph.adj1)
    graph.adj2 = torch.FloatTensor(graph.adj2)
    graph.adj1 = graph.adj1 / graph.adj1.sum(dim=1, keepdim=True)
    graph.adj2 = graph.adj2 / graph.adj2.sum(dim=1, keepdim=True)
    graph.labels = torch.LongTensor(np.array(data['Label']))
    graph.n_nodes = graph.v1.shape[0]
    graph.n_feats = graph.v1.shape[1]
    return graph


def load_flickr_graph():
    graph = Graph()
    data = sio.loadmat('../../data/FLICKR.mat')
    feature = data['Attributes'].todense()
    graph.adj1 = data['Network'].todense() + np.identity(data['Network'].shape[0])
    graph.adj2 = data['Network'].todense() + np.identity(data['Network'].shape[0])
    graph.v1 = sigmoid(feature)
    # # graph.v2 = np.tanh(feature)
    graph.v2 = feature
    graph.train_idx = torch.arange(0, 100)
    graph.test_idx = torch.arange(100, 1000)
    graph.v1 = torch.FloatTensor(graph.v1)
    graph.v2 = torch.FloatTensor(graph.v2)
    graph.v1 = F.normalize(graph.v1, p=2, dim=1)
    graph.v2 = F.normalize(graph.v2, p=2, dim=1)
    graph.adj1 = torch.FloatTensor(graph.adj1)
    graph.adj2 = torch.FloatTensor(graph.adj2)
    graph.adj1 = graph.adj1 / graph.adj1.sum(dim=1, keepdim=True)
    graph.adj2 = graph.adj2 / graph.adj2.sum(dim=1, keepdim=True)
    # nan_idx = graph.v2.isnan()
    # graph.v2[nan_idx] = 0
    graph.labels = torch.LongTensor(np.array(data['Label']))
    graph.n_nodes = graph.v1.shape[0]
    graph.n_feats = graph.v1.shape[1]
    return graph

#
# def load_reddit_graph(name, d):
#     data = torch.load('./data/{}.pt'.format(name))
#     from torch_geometric.utils import to_dense_adj
#     adj = to_dense_adj(data.edge_index).squeeze(dim=0)
#     if not os.path.exists('./data/{}_edge_list.txt'.format(name)):
#         with open('./data/{}_edge_list.txt'.format(name), 'w') as f:
#             for i in range(len(adj)):
#                 idx = np.nonzero(adj[i, :])[0]
#                 for j in idx:
#                     f.write('{} {}\n'.format(i, j))
#     if not os.path.exists('./data/{}_edge_list_emb_{}'.format(name, d)):
#         os.system("python ./deepwalk/main.py --representation-size {} --input ./data/{}_edge_list.txt"
#                   " --output ./data/{}_edge_list_emb_{}".format(d, name, name, d))
#     feature = np.zeros((adj.shape[0], d))
#     with open('./data/{}_edge_list_emb_{}'.format(name, d), 'r') as f:
#         next(f)
#         for line in f.readlines():
#             line = list(map(float, line.split()))
#             feature[int(line[0])] = line[1:]
#     graph = Graph()
#     # feature = data.x
#     graph.adj1 = adj + np.identity(adj.shape[0])
#     graph.adj2 = adj + np.identity(adj.shape[0])
#     graph.v1 = sigmoid(feature)
#     graph.v2 = np.tanh(feature)
#     graph.train_idx = torch.arange(0, 100)
#     graph.test_idx = torch.arange(100, 1000)
#     graph.v1 = torch.FloatTensor(graph.v1)
#     graph.v2 = torch.FloatTensor(graph.v2)
#     # graph.v1 = F.normalize(graph.v1, p=2, dim=1)
#     # graph.v2 = F.normalize(graph.v2, p=2, dim=1)
#     graph.adj1 = graph.adj1.float()
#     graph.adj2 = graph.adj2.float()
#     graph.adj1 = graph.adj1 / graph.adj1.sum(dim=1, keepdim=True)
#     graph.adj2 = graph.adj2 / graph.adj2.sum(dim=1, keepdim=True)
#     label = data.y
#     label[label != 0] = 1
#     graph.labels = label.long()
#     graph.n_nodes = graph.v1.shape[0]
#     graph.n_feats = graph.v1.shape[1]
#     return graph

