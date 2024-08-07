import torch
import numpy as np
import argparse
import time
from Model.Trainer import Trainer
model_config = {
    'data_dim': 8,
    'epochs': 10,
    'learning_rate': 0.01,  # 0.1, 0.05, 0.01, 0.005 ,0.001, 0.0001
    'sche_gamma': 0.98,
    'mask_num': 5,
    'lambda': 5, # 1, 5, 10, 20, 50, 100
    'device': 'cuda:0',
    'data_dir': 'Data/',
    'runs': 20,
    'batch_size': 512, 
    'en_nlayers': 3,
    'de_nlayers': 3,
    'hidden_dim': 32,
    'z_dim': 32,
    'mask_nlayers': 3,
    'random_seed': 1234,
    'num_workers': 0
}

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
    parser.add_argument('--ratio', type=float, default=0)

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
    torch.manual_seed(model_config['random_seed'])
    torch.cuda.manual_seed(model_config['random_seed'])
    np.random.seed(model_config['random_seed'])
    if model_config['num_workers'] > 0:
        torch.multiprocessing.set_start_method('spawn')
    result = []
    runs = model_config['runs']
    mse_rauc, mse_ap, mse_f1 = np.zeros(runs), np.zeros(runs), np.zeros(runs)
    trainer = Trainer(args=args, model_config=model_config)
    start_time = time.time()
    for i in range(runs):
        trainer.training(model_config['epochs'])
        print("Running epoch:", i)
        trainer.evaluate(mse_rauc, mse_ap, mse_f1)
        end_time = time.time()
        print("Time used:", end_time - start_time)
    # mean_mse_auc , mean_mse_pr , mean_mse_f1 = np.mean(mse_rauc), np.mean(mse_ap), np.mean(mse_f1)
    #
    # print('##########################################################################')
    # print("mse: average AUC-ROC: %.4f  average AUC-PR: %.4f"
    #       % (mean_mse_auc, mean_mse_pr))
    # print("mse: average f1: %.4f" % (mean_mse_f1))
    # results_name = './results/' + model_config['dataset_name'] + '.txt'
    #
    # with open(results_name,'a') as file:
    #     file.write("epochs: %d lr: %.4f gamma: %.2f masks: %d lambda: %.1f " % (
    #         model_config['epochs'], model_config['learning_rate'], model_config['sche_gamma'], model_config['mask_num'], model_config['lambda']))
    #     file.write('\n')
    #     file.write("de_layer: %d  hidden_dim: %d z_dim: %d mask_layer: %d" % (model_config['de_nlayers'], model_config['hidden_dim'], model_config['z_dim'], model_config['mask_nlayers']))
    #     file.write('\n')
    #     file.write("mse: average AUC-ROC: %.4f  average AUC-PR: %.4f average f1: %.4f" % (
    #         mean_mse_auc, mean_mse_pr, mean_mse_f1))
    #     file.write('\n')