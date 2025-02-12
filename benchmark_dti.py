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
from trainer.trainer_dti import DTITrainer
# from geognn import GeoUnetPP
from models.model_dti import UnetDTI
from dataset.databuild_dti import get_benchmark_loader

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', type=str, default='bindingdb',
                        help='Bnechmark dataset name(default: bindingdb)')
    parser.add_argument('--split', type=str, default='random',
                        help='Split mode: random or cluster(default: random)')
    # Add argument
    parser.add_argument('--runs', type=int, default=1,
                        help='indepent run times (default: 5)')
    parser.add_argument('--lr_reduce_rate', default=0.5, type=float,
                        help='learning rate reduce rate (default: 0.5)')
    parser.add_argument('--lr_reduce_patience', default=50, type=int,
                        help='learning rate reduce patience (default: 5)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='maximum training epochs (default: 100)')
    parser.add_argument('--log_train_results', action="store_false",
                        help='whether to evaluate training set in each epoch, costs more time (default: True)')
    args = parser.parse_args()
    
#%%
    os.makedirs('checkpoint/DTI/', exist_ok=True)
    os.makedirs('log/DTI/', exist_ok=True)
    args.min_epochs = 1
    args.patience = 30
    args.lr = 5e-5
    np.random.seed(666779)
    
    config = yaml.load(open('./config.yaml', 'r'), Loader=yaml.CLoader)  
    args.config = json.dumps(config)
    args.mode = 'cls'
    args.monitor = 'rmse' if args.mode == 'reg' else 'roc'
    # args.monitor = 'loss'
    config['model']['hidden_dim'] = 128
    metric_list = []
    test_results = []
    for i in range(args.runs):
        seed = np.random.randint(0, 100000)
        loader_tr, loader_va, loader_te = get_benchmark_loader(args.dataset, args.split, batch_size=64, seed=seed)
        print(f"Dataset size: {len(loader_tr.dataset)} | {len(loader_va.dataset)} | {len(loader_te.dataset)}")
        
        data_list = {d[0].smiles: d[0] for d in loader_tr.dataset}.values()
        config['deg'] = get_deg_from_list(data_list)
        
        set_seed(seed)
        model = UnetDTI(config)

        
        device = torch.device('cuda:0') 
        tr = DTITrainer(args, model, device)
        info_dict = tr(loader_tr, loader_va, loader_te, tensorboard=False, save_path=f'checkpoint/DTI/{args.dataset}.pt')
        
        metric_list.append(info_dict[args.monitor])
        metric_np = np.array(metric_list)
        
        with open(f'log/DTI/{args.dataset}.txt', 'a') as f:
            f.write(f"================= {i+1} run {seed} =================\n")
            f.write("{:.4f}\n".format(metric_list[-1]))
            f.write("{:.4f} +/- {:.4f}\n".format(metric_np.mean(), metric_np.std()))
        
        results = tr.test(loader_te)
        test_results.append(results)
        np.save(f'log/DTI/{args.dataset}.npy', np.array(test_results))

















   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    