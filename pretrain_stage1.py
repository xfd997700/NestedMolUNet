# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 20:28:01 2024

@author: Fanding Xu
"""

import os
import time
import torch
import json
import yaml
import argparse

from utils import set_seed, get_deg_from_list
from models.model_pretrain import UnetPretrainS1
from dataset import get_pretrain_loader
from trainer import PretrainTrainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Add argument
    parser.add_argument('--seed', type=float, default=114514,
                        help='Random seed(default: 114514)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr_reduce_rate', default=0.5, type=float,
                        help='learning rate reduce rate (default: 0.5)')
    parser.add_argument('--lr_reduce_patience', default=3, type=int,
                        help='learning rate reduce patience (default: 3)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--patience', type=int, default=8,
                        help='early stop patience (default: 8)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='maximum training epochs (default: 30)')
    parser.add_argument('--min_epochs', type=int, default=1,
                        help='minimum training epochs (default: 1)')
    parser.add_argument('--log_train_results', action="store_false",
                        help='whether to evaluate training set in each epoch, costs more time (default: False)')
    args = parser.parse_args()

#%%
    dataset = 'zinc2m'
    root = './dataset/data/pretrain/processed'
    load_path = os.path.join(root, dataset+'.pth')
    assert os.path.exists(load_path), "Processed dataset not found, please check the path and run dataset/databuild_pretrain.py first."
    data_list = torch.load(load_path)

    test_name = "pretrain"
    seed = 123

    run_time = time.strftime("S1_RUN-%Y%m%d-%H%M%S", time.localtime())
    log_path = f"log/pretrain/s1_{test_name}_{run_time}.txt"

    device = torch.device('cuda:0')
    config = yaml.load(open('config.yaml', 'r'), Loader=yaml.CLoader)
    config['FP']['attr_fp'] = False
    config['FP']['query_fp'] = False
    args.config = json.dumps(config)
    with open(log_path,"a") as f:
        f.write(args.config+'\n')
    
    loader_tr, loader_va, loader_te = get_pretrain_loader(data_list, batch_size=512, num_workers=0)
    
    config['deg'] = get_deg_from_list(loader_tr.dataset)
    # config['deg'] = torch.tensor([       0,  9040426, 26966933, 16482293,   761158,       11])
    
    loss, roc, prc = [], [], []
    
    set_seed(seed)
    model = UnetPretrainS1(config).to(device)   
    
    tr = PretrainTrainer(args, model, device)

    info_dict = tr(loader_tr, loader_va, loader_te, log_path=log_path,
                   tensorboard=False, save_path=f'checkpoint/pretrain/model_s1_{test_name}.pt')
    
    torch.save(model.unet.state_dict(), f'checkpoint/pretrain/unet_s1_{test_name}.pt')
    

    loss = info_dict['loss']
    separate_loss = info_dict['separate_loss']

    
    print("*****************    F i n i s h e d !    *****************")
    print('loss: {:.4f}'.format(loss))
    print(f'seperate_loss: {separate_loss}') 
        
    with open(log_path,"a") as f:
        info = ['loss: {:.4f}'.format(loss),
                f'seperate_loss: {separate_loss}']
        f.write(f'{info[0]}\n{info[1]}\n')