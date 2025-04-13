# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 16:35:22 2023

@author: Fanding Xu
"""

import pickle
import torch
import os
import numpy as np
from tqdm import tqdm
import yaml
import pandas as pd
from sklearn import preprocessing
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch
from sklearn.model_selection import train_test_split
from tdc import utils
from tdc.benchmark_group import admet_group

from .databuild import labeled_data
from .databuild_pretrain import make_graph_data
# names = utils.retrieve_benchmark_names('ADMET_Group')


class scaler:
    """
    from https://github.com/maplightrx/MapLight-TDC
    """
    def __init__(self, log=False):
        self.log = log
        self.offset = None
        self.scaler = None

    def fit(self, y):
        # make the values non-negative
        self.offset = np.min([np.min(y), 0.0])
        y = y.reshape(-1, 1) - self.offset

        # scale the input data
        if self.log:
            y = np.log10(y + 1.0)

        self.scaler = preprocessing.StandardScaler().fit(y)

    def transform(self, y):
        y = y.reshape(-1, 1) - self.offset

        # scale the input data
        if self.log:
            y = np.log10(y + 1.0)

        y_scale = self.scaler.transform(y)

        return y_scale

    def inverse_transform(self, y_scale):
        y = self.scaler.inverse_transform(y_scale.reshape(-1, 1))

        if self.log:
            y = 10.0**y - 1.0

        y = y + self.offset

        return y


class DatasetConfig:
    config_dict = {'caco2_wang': {'task': 'reg', 'monitor': 'mae', 'log_scale': False},
                   'hia_hou': {'task': 'cls', 'monitor': 'roc', 'log_scale': False},
                   'pgp_broccatelli': {'task': 'cls', 'monitor': 'roc', 'log_scale': False},
                   'bioavailability_ma': {'task': 'cls', 'monitor': 'roc', 'log_scale': False},
                   'lipophilicity_astrazeneca': {'task': 'reg', 'monitor': 'mae', 'log_scale': False},
                   'solubility_aqsoldb': {'task': 'reg', 'monitor': 'mae', 'log_scale': False},
                   'bbb_martins': {'task': 'cls', 'monitor': 'roc', 'log_scale': False},
                   'ppbr_az': {'task': 'reg', 'monitor': 'mae', 'log_scale': False},
                   'vdss_lombardo': {'task': 'reg', 'monitor': 'spearman', 'log_scale': True},
                   'cyp2d6_veith': {'task': 'cls', 'monitor': 'prc', 'log_scale': False},
                   'cyp3a4_veith': {'task': 'cls', 'monitor': 'prc', 'log_scale': False},
                   'cyp2c9_veith': {'task': 'cls', 'monitor': 'prc', 'log_scale': False},
                   'cyp2d6_substrate_carbonmangels': {'task': 'cls', 'monitor': 'prc', 'log_scale': False},
                   'cyp3a4_substrate_carbonmangels': {'task': 'cls', 'monitor': 'prc', 'log_scale': False},
                   'cyp2c9_substrate_carbonmangels': {'task': 'cls', 'monitor': 'prc', 'log_scale': False},
                   'half_life_obach': {'task': 'reg', 'monitor': 'spearman', 'log_scale': True},
                   'clearance_microsome_az': {'task': 'reg', 'monitor': 'spearman', 'log_scale': True},
                   'clearance_hepatocyte_az': {'task': 'reg', 'monitor': 'spearman', 'log_scale': True},
                   'herg': {'task': 'cls', 'monitor': 'roc', 'log_scale': False},
                   'ames': {'task': 'cls', 'monitor': 'roc', 'log_scale': False},
                   'dili': {'task': 'cls', 'monitor': 'roc', 'log_scale': False},
                   'ld50_zhu': {'task': 'reg', 'monitor': 'mae', 'log_scale': False}
                   }
    
    @staticmethod
    def register_args(args):
        names = DatasetConfig.config_dict.keys()
        try:
            dataset = args.dataset
        except: raise ValueError("dataset name not defined")
        assert dataset in names, f"wrong dataset name: {dataset}, check in {names}"
        args.task = DatasetConfig.config_dict[dataset]['task']
        args.monitor = DatasetConfig.config_dict[dataset]['monitor']
        args.log_scale = DatasetConfig.config_dict[dataset]['log_scale']
        
        args.num_class = 1
        
        # args.lr = 1e-4
        # args.min_epochs = 30
        # args.patience = 10


def get_mol_dict(group, name):
    save_path = os.path.join(group.path, f'../{name}.pth')
    if os.path.exists(save_path):
        print(f"Loading exist mol_dict {save_path}...")
        id2mol_dict = torch.load(save_path)
        print("Done!")
        return id2mol_dict
    
    benchmark = group.get(name)
    train_val, test = benchmark['train_val'], benchmark['test']
    # id2mol_dict = {}''
    train_val_dict = {}
    for index, row in tqdm(train_val.iterrows(), total=len(train_val),
                           desc="generating mol graph data..."):
        drug_id = row['Drug_ID']
        smiles = row['Drug']
        data = make_graph_data(smiles,
                               get_desc=True, get_fp=True,
                               with_hydrogen=False,
                               with_coordinate=False, seed=123)
        data.y = torch.tensor([[row['Y']]], dtype=torch.float32)

        train_val_dict[drug_id] = data
    
    tset_dict = {}
    for index, row in tqdm(test.iterrows(), total=len(test),
                           desc="generating mol graph data..."):
        drug_id = row['Drug_ID']
        smiles = row['Drug']
        data = make_graph_data(smiles,
                               get_desc=True, get_fp=True,
                               with_hydrogen=False,
                               with_coordinate=False, seed=123)
        data.y = torch.tensor([[row['Y']]], dtype=torch.float32)
        
        tset_dict[drug_id] = data
        
    id2mol_dict = {'train_val': train_val_dict,
                   'test': tset_dict}
    print(f"Saving to {save_path}...")
    torch.save(id2mol_dict, save_path)
    print("Done")
    return id2mol_dict

    
class admetDataset(Dataset):
    def __init__(self, df, mol_dict):
        self.mol_dict = mol_dict
        self.df = df
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        data = self.mol_dict[row['Drug_ID']]
        return data



class admetDataLoader(DataLoader):
    def __init__(self, data, **kwargs):
        super().__init__(data, collate_fn=self.collate_fn, **kwargs)

    def collate_fn(self, batch):
        graph_list = []
        for data in batch:
            graph_list.append(data)
        graph = Batch.from_data_list(graph_list)
        return graph
    
def get_admet_loader(group, name, id2mol_dict=None, 
                     split_type='default', batch_size=None,
                     seed=123, num_workers=0):
    if id2mol_dict is None:
        id2mol_dict = get_mol_dict(group, name)
    benchmark = group.get(name)
    if batch_size is None:
        size = len(id2mol_dict['train_val']) + len(id2mol_dict['test'])
        if size <= 5000: batch_size = 32
        elif size <= 10000: batch_size = 64
        else: batch_size = 128
    
    _, test = benchmark['train_val'], benchmark['test']
    train, valid = group.get_train_valid_split(benchmark=name, split_type=split_type, seed=seed)
    
    torch.manual_seed(seed=seed)
    loader_tr = admetDataLoader(admetDataset(train, id2mol_dict['train_val']),
                                batch_size=batch_size, shuffle=True, drop_last=False,
                                num_workers=num_workers)
    loader_va = admetDataLoader(admetDataset(valid, id2mol_dict['train_val']),
                                batch_size=batch_size, shuffle=False, drop_last=False,
                                num_workers=num_workers)
    loader_te = admetDataLoader(admetDataset(test, id2mol_dict['test']),
                                batch_size=batch_size, shuffle=False, drop_last=False,
                                num_workers=num_workers)
    return loader_tr, loader_va, loader_te

    
def easy_loader(train, valid, test, id2mol_dict,
                batch_size=None,
                seed=114514, num_workers=0):
    if batch_size is None:
        size = len(id2mol_dict['train_val']) + len(id2mol_dict['test'])
        if size <= 5000: batch_size = 32
        elif size <= 10000: batch_size = 64
        else: batch_size = 128
        
    torch.manual_seed(seed=seed)
    loader_tr = admetDataLoader(admetDataset(train, id2mol_dict['train_val']),
                                batch_size=batch_size, shuffle=True, drop_last=False,
                                num_workers=num_workers)
    # TODO: 这里做了修改
    loader_va = admetDataLoader(admetDataset(valid, id2mol_dict['train_val']),
                                batch_size=int(batch_size*2), shuffle=False, drop_last=False,
                                num_workers=num_workers)
    loader_te = admetDataLoader(admetDataset(test, id2mol_dict['test']),
                                batch_size=int(batch_size*2), shuffle=False, drop_last=False,
                                num_workers=num_workers)
    return loader_tr, loader_va, loader_te




"""
===============================================================================
                               ADMET Pretrain
===============================================================================
"""
from lightning.pytorch.utilities import CombinedLoader

# *****************************************************************************
task_batch_size = {'Toxity': 64,
                   'Dristribution': 16,
                   'Absorption': 16,
                   'Metabolism': 64,
                   'PCBA': 256,
                   'HIV':64}

class PretrainAuxDataset(Dataset):
    def __init__(self, data_list, name=None):
        self.samples = data_list
        self.name = name
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]
    

class PretrainADMETDataset(Dataset):
    def __init__(self, df, mol_dict,
                 task_type={'cls':[], 'reg':[]},
                 name=None):
        self.df = df
        self.task_type = task_type
        
        have_cls = len(task_type['cls']) > 0
        have_reg = len(task_type['reg']) > 0
        
        self.name = name
        samples = []
        for i, row in df.iterrows():
            try:
                data = mol_dict[row['SMILES']].clone()
            except: continue
        
            if have_cls:
                cls_label = row.loc[task_type['cls']].fillna(-1).values
                data.y_cls = torch.tensor([cls_label], dtype=torch.float32)
                
            if have_reg:
                reg_label = row.loc[task_type['reg']].fillna(-20).values
                data.y_reg = torch.tensor([reg_label], dtype=torch.float32)
            
            # data.task_name = self.name
            samples.append(data)
        self.samples = samples
        
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]
    
    def __repr__(self):
        rep = f"PretrainADMETDataset (name: {self.name}, task_type: {self.task_type})"
        return rep

def getMultiTaskDataset():
    print("Loading admet datasets...")
    mol_dict = torch.load('dataset/data/admet/processed/admet.pth')
    
    config = yaml.load(open('dataset/admet_info.yaml', 'r'), Loader=yaml.CLoader)
    root = 'dataset/data/admet_open_data/admet_labelled_data'
    df_dict = {}
    
    dataset_dict_tr = {}
    dataset_dict_va = {}
    dataset_dict_te = {}
    
    for prof, detail in config.items():
        df_dict[prof] = pd.DataFrame(columns=['SMILES'])
        task_type = {'cls':[], 'reg': []}
        for name, task in detail.items():
            task_type[task].append(name)
            cur_df = pd.read_csv(os.path.join(root, f'{name}/data.csv'))
            cur_df.columns = ['SMILES', name]
            df_dict[prof] = pd.merge(df_dict[prof], cur_df, on='SMILES', how='outer')
        
        df_tr, df_te = train_test_split(df_dict[prof], test_size=0.2, random_state=114514)
        df_va, df_te = train_test_split(df_te, test_size=0.5, random_state=114514)
        
        
        dataset_dict_tr[prof] = PretrainADMETDataset(df_tr, mol_dict,
                                                     task_type, prof)
        dataset_dict_va[prof] = PretrainADMETDataset(df_va, mol_dict,
                                                     task_type, prof)
        dataset_dict_te[prof] = PretrainADMETDataset(df_te, mol_dict,
                                                     task_type, prof)
    print("DONE")
    print("Loading HIV dataset...")
    data_list = torch.load('dataset/data/admet/processed/hiv.pth')
    df_tr, df_te = train_test_split(data_list, test_size=0.2, random_state=114514)
    df_va, df_te = train_test_split(df_te, test_size=0.5, random_state=114514)
    dataset_dict_tr['HIV'] = PretrainAuxDataset(df_tr, 'HIV')
    dataset_dict_va['HIV'] = PretrainAuxDataset(df_va, 'HIV')
    dataset_dict_te['HIV'] = PretrainAuxDataset(df_te, 'HIV')
    print("DONE")
    
    print("Loading PCBA dataset...")
    data_list = torch.load('dataset/data/admet/processed/pcba.pth')
    df_tr, df_te = train_test_split(data_list, test_size=0.2, random_state=114514)
    df_va, df_te = train_test_split(df_te, test_size=0.5, random_state=114514)
    dataset_dict_tr['PCBA'] = PretrainAuxDataset(df_tr, 'PCBA')
    dataset_dict_va['PCBA'] = PretrainAuxDataset(df_va, 'PCBA')
    dataset_dict_te['PCBA'] = PretrainAuxDataset(df_te, 'PCBA')
    print("DONE")

    return dataset_dict_tr, dataset_dict_va, dataset_dict_te


def getMultiTaskLoader(batch_size_dict=task_batch_size):
    dict_tr, dict_va, dict_te = getMultiTaskDataset()
    for k in dict_tr.keys():
        dict_tr[k] = admetDataLoader(dict_tr[k], batch_size=batch_size_dict[k], shuffle=True)
    for k in dict_tr.keys():
        dict_va[k] = admetDataLoader(dict_va[k], batch_size=batch_size_dict[k], shuffle=False)
    for k in dict_tr.keys():
        dict_te[k] = admetDataLoader(dict_te[k], batch_size=batch_size_dict[k], shuffle=False)
        
    loader_tr = CombinedLoader(dict_tr, 'max_size_cycle')
    loader_va = CombinedLoader(dict_va, 'max_size_cycle')
    loader_te = CombinedLoader(dict_te, 'max_size_cycle')
    
    return loader_tr, loader_va, loader_te

# *****************************************************************************


  
    















































































