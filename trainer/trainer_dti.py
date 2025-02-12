# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 14:18:37 2023

@author: Fanding Xu
"""

import os
import time
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error, r2_score, roc_auc_score, auc, precision_recall_curve
from sklearn.metrics import average_precision_score, roc_curve, confusion_matrix, precision_score
# from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole, rdMolDraw2D
from torch.utils.tensorboard import SummaryWriter
from IPython.display import display, Image, SVG

import sys
sys.path.append('../')
from dataset.databuild import from_smiles
from utils import mol_with_atom_index, comps_visualize_multi
def aver_list(x):
    return sum(x) / len(x)

class DTITrainer():
    def __init__(self, args, model, device):
        self.args = args
        if args.patience == 0:
            args.epochs = args.min_epochs
        self.model = model.to(device)
        self.device = device
        self.mode = args.mode
        if self.mode == 'reg':
            self.val_all = self.__val_reg
            self.loss_function = torch.nn.MSELoss() 
            scheduler_mode = 'min'
        elif self.mode == 'cls':
            self.val_all = self.__val_cls
            self.loss_function = torch.nn.BCEWithLogitsLoss()
            scheduler_mode = 'max'
        else: raise ValueError
        
        # optimizer_grouped_parameters = [
        #     {'params': model.unet.parameters(),
        #       'weight_decay': 0.0, 'lr': args.lr * 0.2},
        #     {'params': [p for n, p in model.named_parameters() if not n.startswith('unet')],
        #       'weight_decay': 0.0, 'lr': args.lr}
        #     ]
        # self.optimizer = torch.optim.Adam(optimizer_grouped_parameters)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode=scheduler_mode, factor=args.lr_reduce_rate,
                                                                    patience=args.lr_reduce_patience, min_lr=1e-8)
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=args.lr_reduce_patience,
        #                                                  gamma=args.lr_reduce_rate)
        
            
    def load_buffer(self, load_path):
        assert os.path.exists(load_path), "load_path does not exist"
        self.model.load_state_dict(torch.load(load_path), strict=False)
        print(f"********** Model Loaded {load_path} **********")
    
        
    
    def __call__(self, loader_tr, loader_va, loader_te=None,
                 save_path='checkpoint/ec50.pt', load_path=None, tensorboard=False):
        
        if tensorboard:
            tb = SummaryWriter(log_dir='log/tensorboard')
            note = defaultdict(list)
        
        self.run_time = time.strftime("RUN-%Y%m%d-%H%M%S", time.localtime())
        
        with open(f"log/DTI/detail/{self.run_time}.txt","w") as f:
            f.write(self.args.config+'\n')
        if load_path is not None:
            self.load_buffer(load_path)
        
        best = None
        for epoch in range(1, self.args.epochs + 1):
            tic = time.time()
            print("Epoch: {:d}/{:d}".format(epoch, self.args.epochs), end='') 
            self.model.train()
            epoch_loss = 0   
            for g, t, y in tqdm(loader_tr, desc="Epoch: {:d}/{:d}".format(epoch, self.args.epochs)):
                g = g.to(self.device)
                t = t.to(self.device)
                y = y.to(self.device)
                _, pred = self.model(g, t)
                # print(_)
                pred = pred.view(-1)
                loss = self.loss_function(pred, y)
                epoch_loss += loss.item()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                torch.cuda.empty_cache()
                            
            loss_tr = epoch_loss / (len(loader_tr))
            plr = self.optimizer.param_groups[0]['lr']
            print(f'lr：{round(plr, 7)}')
            print("Training loss = {:.4f}".format(loss_tr))
            
            # info_dict = self.val_only_loss(loader_va)
            info_dict = self.val_all(loader_va)
            
            #************************* tensorboard ****************************
            if tensorboard:
                for key, value in info_dict.items():
                    tb.add_scalars(key, {'val': value}, epoch)
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        note[name+'_param'].append(param.mean())
                        note[name+'_grad'].append(param.grad.mean())
                        tb.add_histogram(name + '_param', param, epoch)
                        tb.add_histogram(name + '_grad', param.grad, epoch)
                    else:
                        print(name)
            #******************************************************************
            
            
            self.log_info(info_dict)
            loss_va = info_dict['loss'] 
            # state = loss_va
            state = info_dict[self.args.monitor]
            
            # self.scheduler.step(state)
            # self.scheduler.step()
            print(self.test(loader_te))
            if epoch > self.args.min_epochs:
                if self.args.monitor in ['rmse', 'loss']:
                    judge = (best - state) > 1e-6
                else:
                    judge = (state - best) > 1e-6
                    
                # if best is None or (best - loss_va) > 1e-6:
                if judge:
                    best = state
                    best_epoch = epoch
                    torch.save(self.model.state_dict(), save_path)
                    times = 0
                    print("New best model saved, epoch {:d}, best metric: {:.4f}".format(best_epoch, best))
                    
                    # self.check_pool()
                else:
                    times += 1
                    print("Not improved for {:d}/{:d} times, current best is epoch {:d}: {:.4f}".format(times, self.args.patience, best_epoch, best))
                if times >= self.args.patience:
                    break
                
            else:
                best = state
                best_epoch = epoch
                torch.save(self.model.state_dict(), save_path)
                times = 0
                print("Model saved, epoch {:d}, current metric: {:.4f}".format(epoch, state))
                
            toc = time.time()
            print("Time costs: {:.3f}\n".format(toc-tic))
        
        self.load_buffer(save_path)
        # torch.save(self.model.unet.state_dict(), 'checkpoint/unet.pt')
        if tensorboard:
            tb.close()
        if loader_te is not None:
            info_dict = self.val_all(loader_te)
            self.log_info(info_dict, desc="Test info: ")
            
            
            return info_dict
    
    
    def __val_reg(self, loader):
        self.model.eval()
        y_true = []
        y_scores = []
        loss_val = 0
        for g, t, y in tqdm(loader, desc="validating..."):
            g = g.to(self.device)
            t = t.to(self.device)
            y = y.to(self.device)
            with torch.no_grad():
                _, pred = self.model(g, t)
                pred = pred.view(-1)
                loss = self.loss_function(pred, y)
                loss_val += loss.item()
                y_true.append(y)
                y_scores.append(pred)
                
        loss_val /= len(loader)
        
        y_true = torch.cat(y_true, dim = 0).cpu().numpy()
        y_scores = torch.cat(y_scores, dim = 0).cpu().numpy()
                
        info_dict = {'loss':loss_val}

        # info_dict['rmse'] = float(mean_squared_error(y_true, y_scores, squared=False))
        
        cur_rmse = root_mean_squared_error(y_true, y_scores)
        
        info_dict = {'loss':loss_val}
        info_dict['rmse'] = float(cur_rmse)
        info_dict['mse'] = float(mean_squared_error(y_true, y_scores))
        info_dict['r2'] = float(r2_score(y_true, y_scores))
        info_dict['mae'] = float(mean_absolute_error(y_true, y_scores))
        return info_dict

    def __val_cls(self, loader):
        self.model.eval()
        y_true = []
        y_scores = []
        loss_val = 0
        for g, t, y in tqdm(loader, desc="validating..."):
            g = g.to(self.device)
            t = t.to(self.device)
            y = y.to(self.device)
            with torch.no_grad():
                _, pred = self.model(g, t)
                pred = pred.view(-1)
                loss = self.loss_function(pred, y)
                loss_val += loss.item()
                y_true.append(y)
                y_scores.append(pred.sigmoid())
                
        loss_val /= len(loader)
        
        y_true = torch.cat(y_true, dim = 0).cpu().numpy()
        y_scores = torch.cat(y_scores, dim = 0).cpu().numpy()
                
        info_dict = {'loss':loss_val}

        # info_dict['rmse'] = float(mean_squared_error(y_true, y_scores, squared=False))
        
        cur_roc = roc_auc_score(y_true, y_scores)
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        cur_prc = auc(recall, precision)
        
        info_dict = {'loss':loss_val}
        info_dict['roc'] = float(cur_roc)
        info_dict['prc'] = float(cur_prc)
        return info_dict
    
    
    
    def test(self, loader):
        self.model.eval()
        y_true = []
        y_scores = []
        loss_val = 0
        for g, t, y in tqdm(loader, desc="validating..."):
            g = g.to(self.device)
            t = t.to(self.device)
            y = y.to(self.device)
            with torch.no_grad():
                _, pred = self.model(g, t)
                pred = pred.view(-1)
                loss = self.loss_function(pred, y)
                loss_val += loss.item()
                y_true.append(y)
                y_scores.append(pred.sigmoid())
                
        loss_val /= len(loader)
        
        y_label = torch.cat(y_true, dim = 0).cpu().numpy()
        y_pred = torch.cat(y_scores, dim = 0).cpu().numpy()
        
        auroc = roc_auc_score(y_label, y_pred)
        auprc = average_precision_score(y_label, y_pred)
        
        fpr, tpr, thresholds = roc_curve(y_label, y_pred)
        prec, recall, _ = precision_recall_curve(y_label, y_pred)
        precision = tpr / (tpr + fpr)
        f1 = 2 * precision * tpr / (tpr + precision + 0.00001)
        thred_optim = thresholds[5:][np.argmax(f1[5:])]
        y_pred_s = [1 if i else 0 for i in (y_pred >= thred_optim)]
        cm1 = confusion_matrix(y_label, y_pred_s)
        accuracy = (cm1[0, 0] + cm1[1, 1]) / sum(sum(cm1))
        sensitivity = cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])
        specificity = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])
        precision1 = precision_score(y_label, y_pred_s)
        
        results = [auroc, auprc, accuracy, sensitivity, specificity, precision1]
        return results
        # return auroc, auprc, np.max(f1[5:]), sensitivity, specificity, accuracy, thred_optim, precision1

    
    
    
    def print_info(self, info_dict, desc="Validation info: "):
        info = ""
        for key, value in info_dict.items():
            if type(value) is dict:
                info += f'{key}: '
                for k, v in value.items():
                    info += f'{k}: {v:.4f}  '
                info += '\t'
            else:
                info += f'{key}: {value:.4f}\t'
        print(desc, end='') 
        print(info)
        return info

    def log_info(self, info_dict, desc="Validation info: "):
        info = self.print_info(info_dict, desc)
        info += "\n"
        f = open(f"log/DTI/detail/{self.run_time}.txt","a")
        f.write(info)
        f.close()
        
    def check_pool(self):
        # smiles = 'COC(=O)N[C@H](C(=O)N[C@@H](Cc1ccccc1)[C@@H](O)CN(Cc1ccc(-c2ccccn2)cc1)NC(=O)[C@@H](NC(=O)OC)C(C)(C)C)C(C)(C)C'
        smiles = 'C1=CC=C(C=C1)C2=CC(=O)C3=C(C(=C(C=C3O2)O[C@H]4[C@@H]([C@H]([C@@H]([C@H](O4)C(=O)O)O)O)O)O)O'
        data = from_smiles(smiles)
        data.batch = torch.zeros(data.x.size(0), dtype=torch.long)
        data = data.to(self.device)
        edge_index = data.edge_index
        self.model.eval()
        with torch.no_grad():
            _ = self.model(data)
           
        comps, tars = self.model.unet.info()
        mol = mol_with_atom_index(Chem.MolFromSmiles(smiles))
        imgs = comps_visualize_multi(mol, comps, tars, edge_index)
        for png in imgs:
            display(Image(png))



































































