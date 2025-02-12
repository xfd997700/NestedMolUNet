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
from utils import set_seed, get_dataset_deg
from models import UnetProperty
from dataset import get_pp_loader_scaffold, DatasetConfig, get_pp_loader_random_single
from trainer import PropertyTrainer
from dataset.dataset_admet import get_admet_loader

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Add argument
    parser.add_argument('--seed', type=float, default=114514,
                        help='Random seed(default: 114514)')
    parser.add_argument('--runs', type=int, default=5,
                        help='indepent run times (default: 5)')
    parser.add_argument('--lr_reduce_rate', default=0.5, type=float,
                        help='learning rate reduce rate (default: 0.5)')
    parser.add_argument('--lr_reduce_patience', default=5, type=int,
                        help='learning rate reduce patience (default: 5)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--epochs', type=int, default=500,
                        help='maximum training epochs (default: 500)')
    parser.add_argument('--log_train_results', action="store_false",
                        help='whether to evaluate training set in each epoch, costs more time (default: True)')
    parser.add_argument('--pretrained', action="store_false",
                        help='whether to load pretrained model (default: False)')
    parser.add_argument('--dataset', type=str, required=True, choices=['BBBP', 'bace', 'clintox', 'tox21', 'sider', 'Lipophilicity', 'freesolv', 'esol', 'ld50'],
                        help='the property prediction dataset')
    args = parser.parse_args()
    
#%%
    dataset = args.dataset
    load_pt = args.pretrained
    config = yaml.load(open('./config.yaml', 'r'), Loader=yaml.CLoader)
    
    log_path = f'log/property/{dataset}/'
    if not os.path.exists(log_path): os.makedirs(log_path)
    

    args.dataset = dataset
    DatasetConfig.register_args(args)
    np.random.seed(args.seed)
    run_time = time.strftime("RUN-%Y%m%d-%H%M%S", time.localtime())
    device = torch.device('cuda:0') 
    config['deg']=None
    args.config = json.dumps(config)
    
    file_name = dataset
    save_name = 'temp'
    if load_pt:
        save_name += '_pt'
        
        file_name += '_pretrained'
        
    if config['FP']['query_fp']:
        save_name += '_fp'
        file_name += '_FP'
    save_path=f'checkpoint/property/{save_name}.pt'
    
    txt_path = os.path.join(log_path, f'{file_name}.txt')
    pkl_path = os.path.join(log_path, f'{file_name}.pkl')
    
    with open(txt_path,"w") as f:
        f.write(args.config+'\n')
        
    loss, metric = [], []
    infos = []
     
    for i in range(args.runs):
        seed = np.random.randint(0, 100000)


        loader_tr, loader_va, loader_te = get_pp_loader_scaffold(dataset, batch_size=args.batch_size, seed=seed)
        
        load_path = f'dataset/data/property/processed/{dataset}.pth'
        config['deg'] = get_dataset_deg(load_path)
        
        print(f"Running {dataset} SF fold {i}")
        print(f"Train: {len(loader_tr.dataset)} | Validation: {len(loader_va.dataset)} | Test: {len(loader_te.dataset)}")
        
        set_seed(seed)
        model = UnetProperty(config, args.num_class).to(device)  
        
        # *************** Load pretrained model ***************
        if load_pt:
            args.min_epochs = 1
            task_name = 'unet_s2_pretrain'
            model.unet.load_state_dict(torch.load(f'checkpoint/pretrain/{task_name}.pt'), strict=False)
            print("Pretrained unet loaded")
        # *****************************************************
        
        tr = PropertyTrainer(args, model, device, pretrained=load_pt)
        info_dict = tr(loader_tr, loader_va, loader_te,
                       log_root = os.path.join(log_path, 'detail/'),
                       save_path=save_path,
                       tensorboard=False)
        infos.append(info_dict)
        
        with open(pkl_path, 'wb') as f:
            pickle.dump(infos, f)
            
        loss.append(info_dict['loss'])
    
        with open(txt_path,"a") as f:
            info = json.dumps(info_dict)
            f.write(info+'\n')
        print(f"\n{i+1}'s independent run")
        
        loss_np = np.array(loss);
        print_info = ['loss: {:.4f} +/- {:.4f}\n'.format(loss_np.mean(), loss_np.std())]
        if args.task == 'cls':
            metric.append([info_dict['roc'], info_dict['prc']])
            metric_np = np.array(metric)
            # roc_np = np.array(roc); prc_np = np.array(prc)
            print_info += ['roc: {:.4f} +/- {:.4f}\n'.format(metric_np[:,0].mean(), metric_np[:,0].std()),
                           'prc: {:.4f} +/- {:.4f}\n\n'.format(metric_np[:,1].mean(), metric_np[:,1].std())]
        elif args.task == 'reg':
            metric.append(info_dict['rmse'])
            metric_np = np.array(metric)
            print_info += ['rmse: {:.4f} +/- {:.4f}\n'.format(metric_np.mean(), metric_np.std())]
        text = ''.join(print_info)
        print(text)
    with open(txt_path, "a") as f:
        f.write(text)
