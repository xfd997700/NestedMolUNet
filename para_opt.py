# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 16:11:56 2024

@author: Fanding Xu
"""

import os
import torch
import numpy as np
import json
import yaml
import argparse
import pandas as pd
from collections import defaultdict

from utils import set_seed, get_dataset_deg, GridSearch
from models import UnetProperty
from dataset import get_pp_loader_scaffold, DatasetConfig
from trainer import PropertyTrainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Add argument
    parser.add_argument('--seed', type=float, default=114514,
                        help='Random seed(default: 114514)')
    parser.add_argument('--lr_reduce_rate', default=0.5, type=float,
                        help='learning rate reduce rate (default: 0.5)')
    parser.add_argument('--lr_reduce_patience', default=5, type=int,
                        help='learning rate reduce patience (default: 5)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--epochs', type=int, default=500,
                        help='maximum training epochs (default: 500)')
    parser.add_argument('--log_train_results', action="store_false",
                        help='whether to evaluate training set in each epoch, costs more time (default: False)')
    args = parser.parse_args()
#%%
    config = yaml.load(open('config.yaml', 'r'), Loader=yaml.CLoader)
    test_set = ['BBBP', 'bace', 'clintox', 'tox21', 'sider', 'esol', 'freesolv', 'Lipophilicity']
    # test_set = ['BBBP', 'bace', 'clintox', 'tox21', 'sider']
    folds = 5

    device = torch.device('cuda:0')
    
    # opt_item = 'MP'
    # opt_space = {}
    # opt_space['method'] = ['GraphTransformer', 'GCN', 'GAT', 'GIN', 'PNA', 'GraphSAGE', 'SuperGAT']

    opt_item = 'predict'
    opt_space = {}
    opt_space['jk'] = ['cat', 'sum', 'score']
    opt_space['graph_pool'] = ['sum', 'attention', 'set2set2', 'ST']
    
    save_root = f'log/para_opt/{opt_item}'
    log_root = f'log/para_opt/{opt_item}/detail'
    if not os.path.exists(save_root): os.makedirs(save_root)
    if not os.path.exists(log_root): os.makedirs(log_root)
    
    gs = GridSearch(opt_space)
    single_test_results = defaultdict(dict)
    with open(os.path.join(save_root, 'grid.txt'), 'w') as f:
        f.write(json.dumps(list(enumerate(gs.grid))))

    for dataset in test_set:
        args.dataset = dataset
        DatasetConfig.register_args(args)
        # args.min_epochs = 1
        
        result_path = os.path.join(save_root, f'{dataset}.csv')
        if args.task == 'cls': columns=['roc_mean', 'roc_std', 'prc_mean', 'prc_std']
        elif args.task == 'reg': columns=['rmse_mean', 'rmse_std']
        result_df = pd.DataFrame(columns=columns,
                                 index=range(len(gs.grid)))
        
        
        for i, params in enumerate(gs.grid):
            np.random.seed(args.seed)
            
            for k, v in zip(opt_space.keys(), params):
                config[opt_item][k] = v
                
            args.config = {k: v for k, v in config.items() if k!='deg'}
            args.config = json.dumps(args.config)
            log_name = f"{dataset}-para_{i}.txt"
            log_path = os.path.join(log_root, log_name)
            with open(log_path,"w") as f:
                f.write(args.config+'\n')

            loss, metric = [], []
            for j in range(folds):
                seed = np.random.randint(0, 100000)
                loader_tr, loader_va, loader_te = get_pp_loader_scaffold(dataset, batch_size=args.batch_size, seed=seed)
                load_path = f'dataset/data/property/processed/{dataset}.pth'
                config['deg'] = get_dataset_deg(load_path)
                
                set_seed(seed)
                model = UnetProperty(config, args.num_class).to(device)    
                
                # task_name = 'cat_sum'
                # model.unet.load_state_dict(torch.load(f'checkpoint/pretrain/unet_s2_{task_name}.pt'), strict=False)
                
                tr = PropertyTrainer(args, model, device)
                info_dict = tr(loader_tr, loader_va, loader_te, tensorboard=False)
                loss.append(info_dict['loss'])
                
                with open(log_path,"a") as f:
                    info = json.dumps(info_dict)
                    f.write(f"run {j+1}" + info + '\n')
                
                loss_np = np.array(loss);
                print_info = ['loss: {:.4f} +/- {:.4f}\n'.format(loss_np.mean(), loss_np.std())]
                if args.task == 'cls':
                    metric.append([info_dict['roc'], info_dict['prc']])
                    metric_np = np.array(metric)
                    # roc_np = np.array(roc); prc_np = np.array(prc)
                    print_info += ['roc: {:.4f} +/- {:.4f}\n'.format(metric_np[:,0].mean(), metric_np[:,0].std()),
                                   'prc: {:.4f} +/- {:.4f}\n\n'.format(metric_np[:,1].mean(), metric_np[:,1].std())]
                    
                    logs = [metric_np[:,0].mean(), metric_np[:,0].std(),
                            metric_np[:,1].mean(), metric_np[:,1].std()]
                elif args.task == 'reg':
                    metric.append(info_dict['rmse'])
                    metric_np = np.array(metric)
                    print_info += ['rmse: {:.4f} +/- {:.4f}\n'.format(metric_np.mean(), metric_np.std())]
                    logs = [metric_np.mean(), metric_np.std()]
                text = ''.join(print_info)
                print(text)
                
                result_df.loc[i] = logs
                single_test_results[dataset][i] = logs
                
            with open(log_path, "a") as f:
                f.write(text)
            
            
            print(result_df, end='\n')
            result_df.to_csv(result_path)
            
            
            
#%%          
    def add_row_data(single_test_results):
        for i, dataset in enumerate(test_set):
            df = pd.read_csv(f'log/para_opt/{opt_item}/{dataset}.csv', index_col=0)
            for k, v in single_test_results[dataset].items():
                df.loc[7] = v
            df.to_csv(f'log/para_opt/{opt_item}/{dataset}.csv')            
            
            
# 0~3 sum query
# 4 cat query lin-batchnorm-relu  
# 5 cat query lin-layernorm-relu        
# 6 cat query lin
# 7 cat query lin-relu    
            
#%%       
    for k in single_test_results.keys():
        path = f'log/para_opt/{opt_item}/{k}.csv'
        data = single_test_results[k]
        data = list(data.values())
        df = pd.DataFrame(data, columns=['roc_mean', 'roc_std', 'prc_mean', 'prc_std'], index=range(len(gs.grid)))
        df.to_csv(path)
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            