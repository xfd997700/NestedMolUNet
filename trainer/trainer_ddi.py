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
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve, f1_score, average_precision_score, accuracy_score

from scipy.stats import spearmanr
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

def accuracy(y_true, y_pred):
    sample_count = 1.
    for s in y_true.shape:
        sample_count *= s

    return np.sum((y_true == y_pred)) / sample_count

class DDITrainer():
    def __init__(self, args, model, device, pretrained=False):
        self.args = args
        if args.patience == 0:
            args.epochs = args.min_epochs
        self.model = model.to(device)
        self.device = device
        self.scheduler_mode = 'max'
        
        self.pretrained = pretrained
        if pretrained:
            optimizer_grouped_parameters = [
                {'params': model.unet.parameters(),
                  'weight_decay': 0.0, 'lr': args.lr * args.lr_multiplier},
                {'params': [p for n, p in model.named_parameters() if not n.startswith('unet')],
                  'weight_decay': 0.0, 'lr': args.lr}
                ]
            self.optimizer = torch.optim.Adam(optimizer_grouped_parameters)     
        
        else:
            self.optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
        
        print(self.optimizer)

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode=self.scheduler_mode, factor=args.lr_reduce_rate,
                                                                    patience=args.lr_reduce_patience, min_lr=1e-8)
        
            
    def load_buffer(self, load_path):
        assert os.path.exists(load_path), "load_path does not exist"
        self.model.load_state_dict(torch.load(load_path), strict=False)
        print(f"********** Model Loaded {load_path} **********")
    
    
    def __call__(self, loader_tr, loader_va, loader_te=None,
                 save_path='checkpoint/temp.pt',
                 log_root = 'log/temp',
                 load_path=None, tensorboard=False):
        
        if tensorboard:
            tb = SummaryWriter(log_dir='log/tensorboard')
            note = defaultdict(list)
        
        self.run_time = time.strftime(f"{self.args.dataset}-%Y%m%d-%H%M%S", time.localtime())
        if not os.path.exists(log_root): os.makedirs(log_root)
        self.log_path = os.path.join(log_root, f"{self.run_time}.txt")
        with open(self.log_path, "w") as f:
            f.write(self.args.config+'\n')
        if load_path is not None:
            self.load_buffer(load_path)
        
        best = None
        for epoch in range(1, self.args.epochs + 1):
            tic = time.time()
            print("Epoch: {:d}/{:d}".format(epoch, self.args.epochs), end='') 
            self.model.train()
            epoch_loss = 0   
            for data in tqdm(loader_tr, desc="Epoch: {:d}/{:d}".format(epoch, self.args.epochs)):
                data = [d.to(self.device) for d in data]
                pred, loss = self.model(data)
                
                epoch_loss += loss.item()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                torch.cuda.empty_cache()
                            
            loss_tr = epoch_loss / (len(loader_tr))
            plr = self.optimizer.param_groups[0]['lr']
            print(f'lr：{round(plr, 7)}')
            print("Training loss = {:.4f}".format(loss_tr))
            

            info_dict = self.val_all(loader_va)
            info_dict['epoch'] = epoch
            print(self.val_all(loader_te))
            
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
            self.scheduler.step(state)
            # self.scheduler.step()
            
            if epoch > self.args.min_epochs:
                if self.scheduler_mode == 'min':
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
                # self.check_pool()
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
            self.log_info(info_dict, desc="Test info: ", test=True)
            return info_dict
    
        
    def val_all(self, loader):
        self.model.eval()
        y_true = []
        y_scores = []
        loss_val = 0
        for data in tqdm(loader, desc="validating..."):
            data = [d.to(self.device) for d in data]
            with torch.no_grad():
                pred, loss = self.model(data)
                y = data[-1]
                loss_val += loss.item()
                y_true.append(y)
                y_scores.append(pred.sigmoid())
                
        loss_val /= len(loader)
        
        y_true = torch.cat(y_true, dim = 0).cpu().numpy()
        y_scores = torch.cat(y_scores, dim = 0).cpu().numpy()
        
        
        roc = roc_auc_score(y_true, y_scores)
        
        pred = (y_scores >= 0.5).astype(int)
        
        acc = accuracy_score(y_true, pred)
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        prc = auc(recall, precision)
        f1 = f1_score(y_true, pred)
        ap = average_precision_score(y_true, pred)
        
        info_dict = {'epoch':-1,
                     'loss':loss_val}
        info_dict['roc'] = roc
        info_dict['prc'] = prc
        info_dict['acc'] = acc
        info_dict['f1'] = f1
        info_dict['ap'] = ap
        
        return info_dict
    
    @torch.no_grad()
    def predict(self, loader):
        preds = []
        for data in tqdm(loader, desc="predicting..."):
            data = data.to(self.device)
            _, pred = self.model(data)
            preds.append(pred)
        preds = torch.cat(preds, dim=0).cpu().numpy()
        return preds
    
    def print_info(self, info_dict, desc="Validation info: "):
        info = ""
        for key, value in info_dict.items():
            if key=='epoch':
                info += f'{key}: {value}\t'
            elif type(value) is dict:
                info += f'{key}: '
                for k, v in value.items():
                        info += f'{k}: {v:.4f}  '
                info += '\t'
            else:
                info += f'{key}: {value:.4f}\t'
        print(desc, end='') 
        print(info)
        return info

    def log_info(self, info_dict, desc="Validation info: ", test=False):
        info = self.print_info(info_dict, desc)
        info += "\n"
        f = open(self.log_path, "a")
        if test:
            f.write('======== Test set result ========\n')
        f.write(info)
        f.close()
        
    def check_pool(self):
        # smiles = 'COC(=O)N[C@H](C(=O)N[C@@H](Cc1ccccc1)[C@@H](O)CN(Cc1ccc(-c2ccccn2)cc1)NC(=O)[C@@H](NC(=O)OC)C(C)(C)C)C(C)(C)C'
        # smiles = 'C1=CC=C(C=C1)C2=CC(=O)C3=C(C(=C(C=C3O2)O[C@H]4[C@@H]([C@H]([C@@H]([C@H](O4)C(=O)O)O)O)O)O)O'
        
        t_smiles = 'CCCc1nn(C)c2c(=O)nc(-c3cc(S(=O)(=O)N4CCN(CC)CC4)ccc3OCC)[nH]c12'
        h_smiles = 'O=[N+]([O-])OCC(CO[N+](=O)[O-])O[N+](=O)[O-]'
        # t = 'Nc1ccc(C(=O)O)cc1'
        # h = 'O=[N+]([O-])O[C@H]1CO[C@H]2[C@@H]1OC[C@H]2O[N+](=O)[O-]'
        
        t = from_smiles(t_smiles, 
                        get_fp=True, 
                        with_hydrogen=False, with_coordinate=True,
                        seed=123, use_OB=True)
        
        h = from_smiles(h_smiles, 
                        get_fp=True, 
                        with_hydrogen=False, with_coordinate=True,
                        seed=123, use_OB=True)
        
        h.batch = torch.zeros(h.x.size(0), dtype=torch.long)
        t.batch = torch.zeros(t.x.size(0), dtype=torch.long)
        
        h = h.to(self.device)
        t = t.to(self.device)

        self.model.eval()
        with torch.no_grad():
            attn = self.model.visual_read(h, t)
           
        pool_info = self.model.unet.pool_info
        
        comps = pool_info[0]['comp']
        tars = pool_info[0]['edge_target']
        h_mol = mol_with_atom_index(Chem.MolFromSmiles(h_smiles))
        imgs = comps_visualize_multi(h_mol, comps, tars, h.edge_index)
        for png in imgs:
            display(Image(png))
            
        comps = pool_info[1]['comp']
        tars = pool_info[1]['edge_target']
        t_mol = mol_with_atom_index(Chem.MolFromSmiles(t_smiles))
        imgs = comps_visualize_multi(t_mol, comps, tars, t.edge_index)
        for png in imgs:
            display(Image(png))

        return attn

    def __repr__(self):
        info = f"device: {self.device}; pretrained: {self.pretrained}"
        return info































































