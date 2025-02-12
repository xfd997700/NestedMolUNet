# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 13:25:13 2024

@author: Fanding Xu
"""

import os
import time
import torch
import pickle
import numpy as np
import json
import yaml
import argparse
from utils import set_seed, get_deg_from_list
from trainer.trainer_ddi import DDITrainer
# from geognn import GeoUnetPP
from models.model_ddi import UnetDDI_SA, UnetDDI_SSI
from dataset.databuild_ddi import get_benchmark_loader

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', type=str, default='drugbank',
                        help='Bnechmark dataset name (default: drugbank)')
    parser.add_argument('--split', type=str, default='S1', choices=['S1', 'S2'],
                        help='Split mode: random or cluster(default: random)')
    # Add argument
    parser.add_argument('--runs', type=int, default=1,
                        help='indepent run times (default: 5)')
    parser.add_argument('--lr_reduce_rate', default=0.5, type=float,
                        help='learning rate reduce rate (default: 0.5)')
    parser.add_argument('--lr_reduce_patience', default=3, type=int,
                        help='learning rate reduce patience (default: 3)')
    parser.add_argument('--decay', type=float, default=5e-4,
                        help='weight decay (default: 0)')
    parser.add_argument('--epochs', type=int, default=30,
                        help='maximum training epochs (default: 30)')
    parser.add_argument('--log_train_results', action="store_false",
                        help='whether to evaluate training set in each epoch, costs more time (default: True)')
    args = parser.parse_args()
    
#%%
    os.makedirs('checkpoint/DDI/', exist_ok=True)
    os.makedirs('log/DDI/', exist_ok=True)
    args.min_epochs = 1
    args.patience = 6
    args.lr = 1e-4
    np.random.seed(114514)
    
    config = yaml.load(open('./config.yaml', 'r'), Loader=yaml.CLoader)  
    args.config = json.dumps(config)
    args.mode = 'cls'
    args.monitor = 'rmse' if args.mode == 'reg' else 'roc'
    
    metric_list = []
    test_results = []
    
    load_pt = False
    config['model']['num_pool_layer'] = 4
    config['model']['hidden_dim'] = 64
    for i in range(args.runs):
        seed = np.random.randint(0, 100000)
        loader_tr, loader_va, loader_te = get_benchmark_loader(args.dataset, args.split, batch_size=128, seed=seed)

        data_list = loader_tr.dataset.unique_data_list()
        config['deg'] = get_deg_from_list(data_list)

        set_seed(seed)
        model = UnetDDI_SSI(config)
        
        device = torch.device('cuda:0') 
        tr = DDITrainer(args, model, device, pretrained=load_pt)
        info_dict = tr(loader_tr, loader_va, loader_te, tensorboard=False,
                       save_path=f'checkpoint/DDI/{args.dataset}.pt')
        
        metric_list.append(info_dict[args.monitor])
        
        metric_np = np.array(metric_list)
        with open(f'log/DDI/{args.dataset}_{args.split}.txt', 'a') as f:
            f.write(f"================= {i+1} run {seed} =================\n")
            f.write("{:.4f}\n".format(metric_list[-1]))
            f.write("{:.4f} +/- {:.4f}\n".format(metric_np.mean(), metric_np.std()))

        test_results.append(info_dict)
        np.save(f'log/DDI/{args.dataset}_{args.split}.npy', np.array(test_results))









   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    