# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 21:07:53 2024

@author: Fanding Xu
"""

import os
import time
import torch
import json
import yaml
import argparse
from utils import set_seed, get_deg_from_list
from models import UnetPretrainS2
from dataset import get_pretrain_loader
from trainer import PretrainTrainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Add argument
    parser.add_argument('--seed', type=float, default=114514,
                        help='Random seed(default: 114514)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--lr_reduce_rate', default=0.5, type=float,
                        help='learning rate reduce rate (default: 0.5)')
    parser.add_argument('--lr_reduce_patience', default=4, type=int,
                        help='learning rate reduce patience (default: 4)')
    parser.add_argument('--decay', type=float, default=1e-4,
                        help='weight decay (default: 0)')
    parser.add_argument('--patience', type=int, default=15,
                        help='early stop patience (default: 12)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='maximum training epochs (default: 100)')
    parser.add_argument('--min_epochs', type=int, default=1,
                        help='the model must train at least min_epoch times (default: 1)')
    parser.add_argument('--log_train_results', action="store_false",
                        help='whether to evaluate training set in each epoch, costs more time (default: False)')
    args = parser.parse_args()

#%%
    dataset = 'chembl'
    root = './dataset/data/pretrain/processed'
    load_path = os.path.join(root, dataset+'.pth')
    assert os.path.exists(load_path), "Processed dataset not found, please check the path and run databuild.py first."
    data_list = torch.load(load_path)

    
    test_name = "pretrain"
    run_time = time.strftime(f"S2_{test_name}_RUN-%Y%m%d-%H%M%S", time.localtime())
    log_path = f"log/pretrain/{run_time}.txt"


    device = torch.device('cuda:0')
    config = yaml.load(open('config.yaml', 'r'), Loader=yaml.CLoader)
    config['FP']['query_fp'] = False
    args.config = json.dumps(config)
    with open(log_path,"a") as f:
        f.write(args.config+'\n')
    
    set_seed(args.seed)
    
    loader_tr, loader_va, loader_te = get_pretrain_loader(data_list, batch_size=512)

    config['deg'] = get_deg_from_list(loader_tr.dataset)
    
    loss, roc, prc = [], [], []
    # mol_dict = torch.load('dataset/data/pretrain/processed/chembl_mol_dict.pth')
    mol_dict = None
    model = UnetPretrainS2(config, mol_dict).to(device)
    stage_load_path = 'checkpoint/pretrain/unet_s1_pretrain.pt'
    model.unet.load_state_dict(torch.load(stage_load_path))
    
    tr = PretrainTrainer(args, model, device)
    
    info_dict = tr(loader_tr, loader_va, loader_te, log_path=log_path,
                   tensorboard=False, save_path=f'checkpoint/pretrain/model_s2_{test_name}.pt')
    
    torch.save(model.unet.state_dict(), f'checkpoint/pretrain/unet_s2_{test_name}.pt')
    
    if hasattr(model, 'attr_decoder'):
        torch.save(model.attr_decoder.state_dict(), f'checkpoint/pretrain/attr_decoder_s2_{test_name}.pt')
        
    if config['pretrain']['graph_pool'] == 'ST':
        torch.save(model.pool.state_dict(), f'checkpoint/pretrain/STPool_s2_{test_name}.pt')
        
    loss = info_dict['loss']
    separate_loss = info_dict['separate_loss']

    
    print("*****************    F i n i s h e d !    *****************")
    print('loss: {:.4f}'.format(loss))
    print(f'seperate_loss: {separate_loss}') 
        
    with open(f"log/pretrain/{run_time}.txt","a") as f:
        info = ['loss: {:.4f}'.format(loss),
                f'seperate_loss: {separate_loss}']
        f.write(f'{info[0]}\n{info[1]}\n')
        f.write(stage_load_path)