# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 15:12:06 2023

@author: Fanding Xu
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_undirected, mask_feature
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool, GlobalAttention, Set2Set, SetTransformerAggregation
from .MolUnet import MolUnetEncoder
from .model_property import UnetProperty


class UnetADMET(UnetProperty):
    def forward(self, g):
        batch = g.batch
        fp_hidden=None
        if self.query_fp:
            fp_hidden, fp_attr = self.fpnn(g.desc)
        xs, _, _ = self.unet(g, query=fp_hidden)

        x = self.__do_jk(xs)
        g_embd = self.pool(x, batch)
        g_embd = self.attr_decoder(g_embd)

        g_embd = F.dropout(g_embd, p=self.dropout_rate, training=self.training)
        if self.query_fp:
            fp_attr = F.dropout(fp_attr, p=self.dropout_rate, training=self.training)
            g_embd = torch.cat([g_embd, fp_attr], dim=-1)
        logits = self.predict(g_embd)

        return xs, logits
    
    def __do_jk(self, src):
        if self.jk == "cat":
            x = torch.cat(src, dim = 1)
        elif self.jk == "last":
            x = src[-1]
        elif self.jk == "max":
            x = torch.max(torch.stack(src, dim = 0), dim = 0)
        elif self.jk == "mean":
            x = torch.mean(torch.stack(src, dim = 0), dim = 0)
        elif self.jk == "sum" or self.jk == "add":
            x = torch.sum(torch.stack(src, dim = 0), dim = 0)
        elif self.jk == "score":
            self.jk_score, x = self.score_jk(src)
        else:
            raise ValueError("Invalid JK type.")
        return x


task_type_dict = {'Toxity': {'cls': 12, 'reg':2},
                  'Dristribution': {'cls': 1, 'reg':1},
                  'Absorption': {'cls': 3, 'reg':2},
                  'Metabolism': {'cls': 10},
                  'PCBA': {'cls': 128},
                  'HIV': {'cls': 1}}
# tasks_list = ['Toxity', 'Dristribution', 'Absorption', 'Metabolism', 'PCBA', 'HIV']

class UnetPretrainADMET(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.unet = MolUnetEncoder(config)
        out_dim = config['model']['hidden_dim']
        num_pool_layer = config['model']['num_pool_layer']
        self.jk = config['pretrain']['jk']
        graph_pool = config['pretrain']['graph_pool']
        
        feature_dim = num_pool_layer * out_dim if self.jk == "cat" else out_dim
        if graph_pool == "sum":
            self.pool = global_add_pool
        elif graph_pool == "mean":
            self.pool = global_mean_pool
        elif graph_pool == "max":
            self.pool = global_max_pool
        elif graph_pool == "attention":
            self.pool = GlobalAttention(gate_nn = torch.nn.Linear(feature_dim, 1))
                
        elif graph_pool[:-1] == "set2set":
            set2set_iter = int(graph_pool[-1])
            self.pool = Set2Set(feature_dim, set2set_iter)
            feature_dim = feature_dim * 2
        elif graph_pool == "ST":
            self.pool = SetTransformerAggregation(feature_dim, 2, 1, 1, heads=4)
            feature_dim *= 2
        else:
            raise ValueError("Invalid graph pooling type.")
            
        self.predict_dict = {}
        
        self.loss_func = nn.ModuleDict()
        self.loss_func['cls'] = nn.BCEWithLogitsLoss()
        self.loss_func['reg'] = nn.MSELoss()
        
        self.predict_dict = nn.ModuleDict()
        
        for task, types in task_type_dict.items():
            self.predict_dict[task] = nn.ModuleDict()
            for k, v in types.items():
                self.predict_dict[task][k] = nn.Sequential(
                    nn.Linear(feature_dim, feature_dim),
                    nn.ReLU(),
                    nn.Linear(feature_dim, v))
                
        self.desc_predictor = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 200))
        

    def forward(self, batch_dict):
        losses = []
        # pred_descs = []
        # label_descs = []
        for name, g in batch_dict.items():
            xs, es, xp = self.unet(g)
            x = self.__do_jk(xs)
            h_g = self.pool(x, g.batch)
            
            # pred_descs.append(self.desc_predictor(h_g))
            # label_descs.append(g.desc)
            
            
            if name == 'HIV':
                logit = self.predict_dict[name]['cls'](h_g)
                label = g.y.view(logit.shape)
                cur_loss = self.loss_func['cls'](logit, label)
                losses.append(cur_loss)
            
            elif name == 'PCBA':
                logit = self.predict_dict[name]['cls'](h_g)
                label = g.y.view(logit.shape)
                mask = label >= 0
                cur_loss = self.loss_func['cls'](logit[mask], label[mask])
                losses.append(cur_loss)
                
            else:
                logit_cls = self.predict_dict[name]['cls'](h_g)
                label_cls = g.y_cls.view(logit_cls.shape)
                mask = label_cls >= 0
                cur_loss_cls = self.loss_func['cls'](logit_cls[mask], label_cls[mask])
                losses.append(cur_loss_cls)
                
                # if name not in ['Metabolism', 'Absorption']:
                #     logit_reg = self.predict_dict[name]['reg'](h_g)
                #     label_reg = g.y_reg.view(logit_reg.shape)
                #     mask = label_reg > -20
                #     cur_loss_reg = self.loss_func['reg'](logit_reg[mask], label_reg[mask])
                #     losses.append(cur_loss_reg)
                
                
        # pred_desc = torch.cat(pred_descs, dim=0)
        # label_desc = torch.cat(label_descs, dim=0)
        # nan_mask = torch.isnan(label_desc)
        # cur_loss_desc = self.loss_func['reg'](pred_desc[~nan_mask], label_desc[~nan_mask])
        # losses.append(cur_loss_desc)
        
        loss = torch.stack(losses).sum()
        
        separate_loss = [x.item() for x in losses]
        
        return separate_loss, loss
            


    def __do_jk(self, src):
        if self.jk == "cat":
            src = torch.cat(src, dim = 1)
        elif self.jk == "last":
            src = src[-1]
        elif self.jk == "max":
            src = torch.max(torch.stack(src, dim = 0), dim = 0)
        elif self.jk == "mean":
            src = torch.mean(torch.stack(src, dim = 0), dim = 0)
        elif self.jk == "sum" or self.jk == "add":
            src = torch.sum(torch.stack(src, dim = 0), dim = 0)
        return src
































