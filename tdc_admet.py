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
import pickle
import argparse
from utils import set_seed, get_deg_from_list
from models.model_admet import UnetADMET
from dataset.dataset_admet import get_admet_loader, DatasetConfig, get_mol_dict, easy_loader, scaler
from trainer import PropertyTrainer
from tdc import utils
from tdc.benchmark_group import admet_group
from tdc.multi_pred import DTI
import ast

def parse_list(arg):
    try:
        return ast.literal_eval(arg)
    except (ValueError, SyntaxError):
        raise argparse.ArgumentTypeError("Invalid list format. Use Python-style lists, e.g., '[1, 0]'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Add argument
    parser.add_argument('--seed', type=float, default=114514,
                        help='Global random seed(default: 114514)')
    parser.add_argument('--seeds', type=list, default=[1, 2, 3, 4, 5],
                        help='indepent run times (default: [1, 2, 3, 4, 5])')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch, if not specified, auto set based on dataset size (default: None)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--lr_reduce_rate', default=0.5, type=float,
                        help='learning rate reduce rate (default: 0.5)')
    parser.add_argument('--lr_reduce_patience', default=10, type=int,
                        help='learning rate reduce patience (default: 10)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--patience', type=int, default=50,
                        help='early stop patience (default: 12)')
    parser.add_argument('--epochs', type=int, default=500,
                        help='maximum training epochs (default: 500)')
    parser.add_argument('--min_epochs', type=int, default=1,
                        help='minimum training epochs (default: 1)')
    parser.add_argument('--log_train_results', action="store_false",
                        help='whether to evaluate training set in each epoch, costs more time (default: False)')
    
    parser.add_argument('--search_pt', type=parse_list, default=[False],
                        help='Search space for pretrained model usage (default: [True, False])')
    parser.add_argument('--search_fp', type=parse_list, default=[True, False],
                        help='Search space for fingerprint query (default: [True, False])')
    parser.add_argument('--search_lr', type=parse_list, default=[1e-3, 1e-4],
                        help='Search space for learning rates (default: [1e-3, 1e-4, 5e-5])')
    parser.add_argument('--search_decay', type=parse_list, default=[0, 1e-4],
                        help='Search space for weight decay (default: [0, 1e-4])')
    parser.add_argument('--run_names', type=parse_list, default=None)
    parser.add_argument('--log_dir', type=str, default='pt',)
    parser.add_argument('--hidden_dims', type=parse_list, default=[64, 128, 256])
    parser.add_argument('--num_pool_layer', type=parse_list, default=[2, 3])
    parser.add_argument('--pred_jk', type=parse_list, default=['cat', 'sum'])
    parser.add_argument('--pred_pool', type=parse_list, default=['sum', 'mean', 'attention'])
    
    args = parser.parse_args()
        
# %%
    data_root = 'dataset/data/admet/'
    os.makedirs(data_root, exist_ok=True)
    group = admet_group(path = data_root)
    if args.run_names is not None:
        names = args.run_names
    else:
        names = utils.retrieve_benchmark_names('ADMET_Group')   
    # names = utils.retrieve_benchmark_names('ADMET_Group')
    predictions_list = []
    
    config = yaml.load(open('./config.yaml', 'r'), Loader=yaml.CLoader)
    device = torch.device('cuda:0') 
    log_path = f'log/admet/{args.log_dir}/'
    os.makedirs(log_path, exist_ok=True)
    os.makedirs('log/temp/', exist_ok=True)
    result_path = os.path.join(log_path, 'result.txt')
    temp_ckpt_path = os.path.join(log_path, 'model.pt')
    
    config['FP']['in_dim'] = 200 # use desc as fp instead
    
    # load_pt = True
    # config['FP']['query_fp'] = True
    all_results = {}
    
    search_pt = args.search_pt
    search_fp = args.search_fp
    search_lr = args.search_lr
    search_decay = args.search_decay
    
    
    for dataset in names:
    # for dataset in ['vdss_lombardo', 'half_life_obach', 'clearance_microsome_az', 'clearance_hepatocyte_az']:

        for pt in search_pt:
            load_pt = pt
            for fp in search_fp:
                config['FP']['query_fp'] = fp
                for lr in search_lr:
                    args.lr = lr
                    for decay in search_decay:
                        args.decay = decay

                        for hd in args.hidden_dims:
                            for npl in args.num_pool_layer:
                                for pred_jk in args.pred_jk:
                                    for pred_pool in args.pred_pool:
                                        

                                        cur_list = []
                                        args.dataset = dataset
                                        id2mol_dict = get_mol_dict(group, dataset)
                                        DatasetConfig.register_args(args)
                                        config['deg'] = None

                                        # --------------------------------------------- #
                                        config['model']['hidden_dim'] = hd
                                        config['model']['num_pool_layer'] = npl
                                        config['predict']['jk'] = pred_jk
                                        config['predict']['graph_pool'] = pred_pool
                                        # --------------------------------------------- #

                                        args.config = json.dumps(config)
                                        config['deg'] = get_deg_from_list(id2mol_dict['train_val'].values())
                                        
                                        loss, metric = [], []
                                        infos = []
                                        for seed in [1, 2, 3, 4, 5]:
                                            benchmark = group.get(dataset) 
                                            # all benchmark names in a benchmark group are stored in group.dataset_names
                                            predictions = {}
                                            name = benchmark['name']
                                            train_val, test = benchmark['train_val'], benchmark['test']
                                            train, valid = group.get_train_valid_split(benchmark = name, split_type = 'default', seed = seed)
                                            
                                            # --------------------------------------------- # 
                                            #  Train your model using train, valid, test    #
                                            #  Save test prediction in y_pred_test variable #
                                            if args.log_scale:
                                                Y_scaler = scaler(log=args.log_scale)
                                                Y_scaler.fit(train['Y'].values)
                                                train['Y_scale'] = Y_scaler.transform(train['Y'].values)
                                                valid['Y_scale'] = Y_scaler.transform(valid['Y'].values)
                                                test['Y_scale'] = Y_scaler.transform(test['Y'].values)

                                            batch_size = args.batch_size
                                            loader_tr, loader_va, loader_te, cur_batch_size = easy_loader(train, valid, test, id2mol_dict, batch_size=batch_size)
                                            print(f"\n\nRunning {dataset} seed {seed}")
                                            print(f"Train: {len(loader_tr.dataset)} | Validation: {len(loader_va.dataset)} | Test: {len(loader_te.dataset)}\n\n")
                                            set_seed(seed)
                                            model = UnetADMET(config, args.num_class).to(device)  
                                            
                                            # *************** Load pretrained model ***************
                                            if load_pt:
                                                args.min_epochs = 1
                                                args.lr_multiplier = 1
                                                args.unet_decay = 0
                                                task_name = 'unet_s2_pretrain'
                                                model.unet.load_state_dict(torch.load(f'checkpoint/pretrain/{task_name}.pt'), strict=False)
                                                print("Pretrained unet loaded")
                                            # *****************************************************
                                            
                                            tr = PropertyTrainer(args, model, device, pretrained=False)
                                            info_dict = tr(loader_tr, loader_va, loader_te, tensorboard=False, save_path=temp_ckpt_path)
                                            
                                            loss.append(info_dict['loss'])
                                            loss_np = np.array(loss);
                                            print_info = ['loss: {:.4f} +/- {:.4f}\n'.format(loss_np.mean(), loss_np.std())]
                                            y_pred_test = tr.predict(loader_te)
                                            if args.log_scale:
                                                y_pred_test = Y_scaler.inverse_transform(y_pred_test)
                                                
                                            # --------------------------------------------- #
                                                
                                            predictions[name] = y_pred_test 
                                            cur_list.append(predictions)
                                            
                                            # ------------ info print per seed ------------ # 
                                            result = group.evaluate(predictions)
                                            result_k, result_v = next(iter(result[dataset].items()))
                                            metric.append(result_v)
                                            print_info += ['current {}: {:.4f}\n'.format(result_k, result_v)]
                                            metric_np = np.array(metric)
                                            print_info += ['{}: {:.4f} +/- {:.4f}\n'.format(result_k, metric_np.mean(), metric_np.std())]
                                            text = ''.join(print_info)
                                            print(text)
                                            # --------------------------------------------- #
                                            
                                        
                                        cur_result = group.evaluate_many(cur_list)  
                                        
                                        all_results.update(cur_result)
                                        print('\n\n{}'.format(cur_result))
                                        
                                        d, r = next(iter(cur_result.items()))
                                        info_to_write = f"{d}\t{r[0]}\t{r[1]}\t{pt}\t{fp}\t{lr}\t{decay}\t{cur_batch_size}\t{hd}\t{npl}\t{pred_jk}\t{pred_pool}"
                                        
                                        
                                        with open(result_path, "a") as f:
                                            f.write(info_to_write+'\n')
                                        
                                        with open(os.path.join(log_path, 'result.pkl'), 'wb') as f:
                                            pickle.dump(all_results, f)
    # results = group.evaluate_many(predictions_list)
















































































