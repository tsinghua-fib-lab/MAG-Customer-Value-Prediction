# -*- coding: utf-8 -*-


import numpy as np
import torch
from torch_geometric.data import DataLoader
from torch_geometric import utils
import torch.nn.functional as F
from torch.utils.data import random_split

import argparse
import os
import time
import setproctitle
from train import *
from dataset_processing import CVPDataset
from gcn_model import *
import logging

import sys

def init_logging_and_result(args):
    global Log_dir_name
    global filename
    Log_dir_name = 'Log'
    if not os.path.exists(Log_dir_name):
        os.makedirs(Log_dir_name)
    
    filename = '{}-lr{}-wd{}-enh{}-gnh{}-nh{}-{}'.format(args.model, args.lr, args.weight_decay, args.enh, args.gnh, args.n_hidden, args.en)
    if not os.path.exists(Log_dir_name + '/' + filename):
        logging.basicConfig(filename=Log_dir_name + '/' + filename, level=logging.INFO)
    else:
        print(Log_dir_name + '/' + filename, 'already exists, removing ...')
        os.remove(Log_dir_name + '/' + filename)
        logging.basicConfig(filename=Log_dir_name + '/' + filename, level=logging.INFO)




if __name__ == '__main__':
    # Experiment parameters
    parser = argparse.ArgumentParser(description='Graph convolutional networks for customer value prediction')
    parser.add_argument('-sd', '--seed', type=int, default=4, help='random seed')
    parser.add_argument('-lr', '--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('-wd', '--weight_decay', type=float, default=1e-3, help='weight decay')
    parser.add_argument('-e', '--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('-d', '--dropout_ratio', type=float, default=0.2, help='dropout rate')
    parser.add_argument('-dvs', '--device', type=str, default='cuda', choices=['cuda:2','cuda:1','cuda:0','cpu'])
    parser.add_argument('-cd', '--cuda_device', type=str, default='1', help='cuda device')
    parser.add_argument('-m', '--model', type=str, default='*', help='model')
    parser.add_argument('-nh', '--n_hidden', type=int, default=32, help='number of hidden nodes in each layer of GCN')
    parser.add_argument('-p', '--patience', type=int, default=80, help='Patience')
    parser.add_argument('-heads', '--heads', type=int, default=1, help='number of heads for GAT')
    parser.add_argument('-gnh', '--gnh', type=int, default=32, help='number of gcn hidden layer.')
    parser.add_argument('-en', '--en', type=str, default='test',help='add experiment name.')
    parser.add_argument('-enh','--enh', type=int, default=32, help='number of embedding size.')
    parser.add_argument('-lf','--loss_fn', type=str, default='mse', help='number of embedding size.')
    parser.add_argument('-ew','--edge_weight', type=str, default='weighted', help='weighed edge.')
    parser.add_argument('-es','--early_stop', type=float, default=0.0, help='the point of early stop.')
    args = parser.parse_args()

    init_logging_and_result(args)
    # 设定相关信息
    torch.manual_seed(args.seed)
    if torch.cuda.is_available:
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True   # 每次训练得到相同结果

    setproctitle.setproctitle('xxxxxxxxxx') # 设定程序名

    ############################### Load Data ###############################
    print('------------------------- Loading data -------------------------')
    # create train, val, test set

    train_loader = CVPDataset(season='q1', directed = False , args=args)
    val_loader = CVPDataset(season='q2', directed = False, args=args)
    test_loader = CVPDataset(season='q3', directed = False, args=args)
    
    
    args.num_features = train_loader[0].num_features

    n_trainset = train_loader.num
    n_valset = val_loader.num
    n_testset = test_loader.num
    print("-- n_trainset:{} --".format(n_trainset))
    print("-- n_valset:{} --".format(n_valset))
    print("-- n_testset:{} --".format(n_testset))

    # 对args做一些处理

    for arg in vars(args):
        print(arg, getattr(args, arg))

    
    print(train_loader[0])

    if args.model=='GCN_embedding':
        model = GCN_embedding(args).to(args.device)
    elif args.model=='GAT_embedding':
        model = GAT_embedding(args).to(args.device)
    elif args.model=='GCN_motif':
        model = GCN_motif(args).to(args.device)
    elif args.model=='GCN_motif_sage':
        model = GCN_motif_sage(args).to(args.device)
    elif args.model=='GCN_motif_wl':
        model = GCN_motif_wl(args).to(args.device)
    elif args.model=='GCN_motif_weighted':
        model = GCN_motif_weighted(args).to(args.device)
    elif args.model=='GCN_motif_weighted_multiple':
        model = GCN_motif_weighted_multiple(args).to(args.device)
    elif args.model=='GCN_motif_nodes':
        model = GCN_motif_nodes(args).to(args.device) 
    elif args.model=='GCN_motif_delete':
        model = GCN_motif_delete(args).to(args.device)
    elif args.model=='GCN_motif_self':
        model = GCN_motif_self(args).to(args.device)
    elif args.model=='GCN_motif_gru':
        model = GCN_motif_gru(args).to(args.device)
    elif args.model=='GCN_motif_gn':
        model = GCN_motif_gn(args).to(args.device)
    elif args.model=='GCN_gate':
        model = GCN_gate(args).to(args.device)                        
    else:
        raise NotImplementedError(args.model)

    

    print(model)
    train_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    print('Trainable Parameters:', np.sum([p.numel() for p in train_params]))
    logging.info('Trainable Parameters:{}'.format(np.sum([p.numel() for p in train_params])))


    train(model, train_loader, val_loader, test_loader, args)
    torch.save(model.state_dict(),'./model/{}-lr{}-wd{}-enh{}-gnh{}-nh{}-{}'.format(args.model, args.lr, args.weight_decay, args.enh, args.gnh, args.n_hidden, args.en))
    
    mae_loss, rmse_loss, mape_loss, mspe_loss = test(model,val_loader,args)
    print("--q2-- MAE:{} RMSE:{} MAPE:{} MSPE:{}".format(mae_loss, rmse_loss, mape_loss, mspe_loss))    
    logging.info("--q2-- MAE:{} RMSE:{} MAPE:{} MSPE:{}".format(mae_loss, rmse_loss, mape_loss, mspe_loss))
    
    mae_loss, rmse_loss, mape_loss, mspe_loss = test(model,test_loader,args)
    print("--q3-- MAE:{} RMSE:{} MAPE:{} MSPE:{}".format(mae_loss, rmse_loss, mape_loss, mspe_loss))    
    logging.info("--q3-- MAE:{} RMSE:{} MAPE:{} MSPE:{}".format(mae_loss, rmse_loss, mape_loss, mspe_loss))
    '''
    test_out = test_y(model, test_loader,args)
    np.save('./data/test_'+str(args.model)+'.npy',test_out)
    test_out = test_y(model, val_loader,args)
    np.save('./data/val_'+str(args.model)+'.npy',test_out)
    
    
    
    model_path = './model/{}-lr{}-wd{}-enh{}-gnh{}-nh{}-{}'.format(args.model, args.lr, args.weight_decay, args.enh, args.gnh, args.n_hidden, args.en)
    model.load_state_dict(torch.load(model_path))
    mae_loss, rmse_loss, mape_loss, mrpe_loss = test(model,train_loader,args)
    

    mae_loss, rmse_loss, mape_loss, mrpe_loss = test(model,val_loader,args)
    print("--q2-- MAE:{} RMSE:{} MAPE:{} MRPE:{}".format(mae_loss, rmse_loss, mape_loss, mrpe_loss))    
    logging.info("--q2-- MAE:{} RMSE:{} MAPE:{} MRPE:{}".format(mae_loss, rmse_loss, mape_loss, mrpe_loss))
    mae_loss, rmse_loss, mape_loss, mrpe_loss = test(model,test_loader,args)
    print("--q3-- MAE:{} RMSE:{} MAPE:{} MRPE:{}".format(mae_loss, rmse_loss, mape_loss, mrpe_loss))    
    logging.info("--q3-- MAE:{} RMSE:{} MAPE:{} MRPE:{}".format(mae_loss, rmse_loss, mape_loss, mrpe_loss))


    test_out = test_y(model, test_loader,args)
    np.save('./data/test_'+str(args.model)+'.npy',test_out)
    test_out = test_y(model, val_loader,args)
    np.save('./data/val_'+str(args.model)+'.npy',test_out)
    '''
