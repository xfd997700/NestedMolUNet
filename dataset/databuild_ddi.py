# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 16:56:49 2023

@author: Fanding Xu
"""

import pandas as pd
import os
import pickle
import torch
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
import logging
from collections import defaultdict

def read_pickle(filename):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    print(filename, " successfully loaded")
    return obj

def mol_filter_by_smi(smiles, labels=None):
    new_smiles, mols = [], []
    if labels is not None:
        assert len(labels) == len(smiles)
        new_labels = []
        
    for i, smi in tqdm(enumerate(smiles), total=len(smiles)):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                mols.append(mol)
                new_smiles.append(smi)
                if labels is not None:
                    new_labels.append(labels[i])
        except:
            continue
    if labels is not None:
        return mols, new_smiles, new_labels
    return mols, new_smiles

def _normal_batch( h, t, r, neg_size, data_statistics, drug_ids, args):
    neg_size_h = 0
    neg_size_t = 0
    prob = data_statistics["ALL_TAIL_PER_HEAD"][r] / (data_statistics["ALL_TAIL_PER_HEAD"][r] + 
                                                            data_statistics["ALL_HEAD_PER_TAIL"][r])
    # prob = 2
    for i in range(neg_size):
        if args.random_num_gen.random() < prob:
            neg_size_h += 1
        else:
            neg_size_t +=1
    
    return (_corrupt_ent(data_statistics["ALL_TRUE_H_WITH_TR"][t, r], neg_size_h, drug_ids, args),
            _corrupt_ent(data_statistics["ALL_TRUE_T_WITH_HR"][h, r], neg_size_t, drug_ids, args))  

def _corrupt_ent(positive_existing_ents, max_num, drug_ids, args):
    corrupted_ents = []
    while len(corrupted_ents) < max_num:
        candidates = args.random_num_gen.choice(drug_ids, (max_num - len(corrupted_ents)) * 2, replace=False)
        invalid_drug_ids = np.concatenate([positive_existing_ents, corrupted_ents], axis=0)
        mask = np.isin(candidates, invalid_drug_ids, assume_unique=True, invert=True)
        corrupted_ents.extend(candidates[mask])

    corrupted_ents = np.array(corrupted_ents)[:max_num]
    return corrupted_ents

def load_data_statistics(all_tuples):
    '''
    This function is used to calculate the probability in order to generate a negative. 
    You can skip it because it is unimportant.
    '''
    print('Loading data statistics ...')
    statistics = dict()
    statistics["ALL_TRUE_H_WITH_TR"] = defaultdict(list)
    statistics["ALL_TRUE_T_WITH_HR"] = defaultdict(list)
    statistics["FREQ_REL"] = defaultdict(int)
    statistics["ALL_H_WITH_R"] = defaultdict(dict)
    statistics["ALL_T_WITH_R"] = defaultdict(dict)
    statistics["ALL_TAIL_PER_HEAD"] = {}
    statistics["ALL_HEAD_PER_TAIL"] = {}

    for h, t, r in tqdm(all_tuples, desc='Getting data statistics'):
        statistics["ALL_TRUE_H_WITH_TR"][(t, r)].append(h)
        statistics["ALL_TRUE_T_WITH_HR"][(h, r)].append(t)
        statistics["FREQ_REL"][r] += 1.0
        statistics["ALL_H_WITH_R"][r][h] = 1
        statistics["ALL_T_WITH_R"][r][t] = 1

    for t, r in statistics["ALL_TRUE_H_WITH_TR"]:
        statistics["ALL_TRUE_H_WITH_TR"][(t, r)] = np.array(list(set(statistics["ALL_TRUE_H_WITH_TR"][(t, r)])))
    for h, r in statistics["ALL_TRUE_T_WITH_HR"]:
        statistics["ALL_TRUE_T_WITH_HR"][(h, r)] = np.array(list(set(statistics["ALL_TRUE_T_WITH_HR"][(h, r)])))

    for r in statistics["FREQ_REL"]:
        statistics["ALL_H_WITH_R"][r] = np.array(list(statistics["ALL_H_WITH_R"][r].keys()))
        statistics["ALL_T_WITH_R"][r] = np.array(list(statistics["ALL_T_WITH_R"][r].keys()))
        statistics["ALL_HEAD_PER_TAIL"][r] = statistics["FREQ_REL"][r] / len(statistics["ALL_T_WITH_R"][r])
        statistics["ALL_TAIL_PER_HEAD"][r] = statistics["FREQ_REL"][r] / len(statistics["ALL_H_WITH_R"][r])
    
    print('getting data statistics done!')

    return statistics

def generate_pair_triplets(args, data, drug_ids=[]):
    pos_triplets = []

    for id1, id2, relation in zip(data[args.c_id1], data[args.c_id2],  data[args.c_y]):
        if ((id1 not in drug_ids) or (id2 not in drug_ids)): continue
        # Drugbank dataset is 1-based index, need to substract by 1
        if args.dataset in ('drugbank', ):
            relation -= 1
        pos_triplets.append([id1, id2, relation])

    if len(pos_triplets) == 0:
        raise ValueError('All tuples are invalid.')

    pos_triplets = np.array(pos_triplets)
    data_statistics = load_data_statistics(pos_triplets)
    drug_ids = np.array(drug_ids)

    neg_samples = []
    for pos_item in tqdm(pos_triplets, desc='Generating Negative sample'):
        temp_neg = []
        h, t, r = pos_item[:3]

        if args.dataset == 'drugbank':
            neg_heads, neg_tails = _normal_batch(h, t, r, args.neg_ent, data_statistics, drug_ids, args)
            temp_neg = [str(neg_h) + '$h' for neg_h in neg_heads] + \
                        [str(neg_t) + '$t' for neg_t in neg_tails]
        else:
            existing_drug_ids = np.asarray(list(set(
                np.concatenate([data_statistics["ALL_TRUE_T_WITH_HR"][(h, r)], data_statistics["ALL_TRUE_H_WITH_TR"][(h, r)]], axis=0)
                )))
            temp_neg = _corrupt_ent(existing_drug_ids, args.neg_ent, drug_ids, args)
        
        neg_samples.append('_'.join(map(str, temp_neg[:args.neg_ent])))
    
    df = pd.DataFrame({'Drug1_ID': pos_triplets[:, 0], 
                        'Drug2_ID': pos_triplets[:, 1], 
                        'Y': pos_triplets[:, 2],
                        'Neg samples': neg_samples})
    
    return df

class easy_data:
    def __init__(self, datamaker, get_fp=False, with_hydrogen=False, with_coordinate=False,
                 seed=123):
        self.datamaker = datamaker
        self.get_fp = get_fp
        self.with_hydrogen = with_hydrogen
        self.with_coordinate = with_coordinate
        self.seed = seed
        self.rdkit_fp = False
        self.mask_attr = False

    def process(self, smi, mol):
        data = self.datamaker(smi, mol, get_fp=self.get_fp,
                              with_hydrogen=self.with_hydrogen, with_coordinate=self.with_coordinate, seed=self.seed)
        return data
    
    def __repr__(self):
        info = f"easy_data(rdkit_fp={self.rdkit_fp}, fpSize={self.fpSize}, mask_attr={self.mask_attr})"
        return info

def prepare_drug_data(args):
    data = pd.read_csv(args.dataset_filename, delimiter=args.delimiter)
    smiles_set = set()

    id2data_dict = {}
    smi2id_dict = {}

    for id1, id2, smiles1, smiles2, relation in zip(data[args.c_id1], data[args.c_id2], data[args.c_s1], data[args.c_s2], data[args.c_y]):
        smi2id_dict[smiles1] = id1
        smi2id_dict[smiles2] = id2
        smiles_set.add(smiles1)
        smiles_set.add(smiles2)
    
    mols, smiles = mol_filter_by_smi(smiles_set)

    ed = easy_data(args.datamaker, get_fp=True, with_hydrogen=False, with_coordinate=False, seed=123)
    data_list = list(Parallel(n_jobs=args.num_processor)(delayed(ed.process)(smi, mol)
                                         for smi, mol in tqdm(zip(smiles_set, mols),
                                                              total=len(smiles_set),
                                                              desc='Generating drug data...')))
    for d in data_list:
        smi = d.smiles
        id2data_dict[smi2id_dict[smi]] = d
    torch.save(id2data_dict, os.path.join(args.dirname, f'{args.dataset}.pth'))


def cold_split(args):
    drug_ids = []
    drug_ids = list(torch.load(os.path.join(args.dirname, f'{args.dataset}.pth')).keys())
    
    train_ids, test_ids = train_test_split(drug_ids, test_size=args.test_ratio, random_state=args.seed)
    
    df = pd.read_csv(args.dataset_filename, delimiter=args.delimiter)
    
    train_list, s1_list, s2_list = [], [], []
    for i in tqdm(range(len(df)), desc="Cold splitting..."):
        pair = df.iloc[i]
        id1, id2 = pair[args.c_id1], pair[args.c_id2]
        if id1 in train_ids and id2 in train_ids:
            train_list.append(i)
        elif id1 in test_ids and id2 in test_ids:
            s1_list.append(i)
        else:
            s2_list.append(i)
    
    df_tr = df.iloc[train_list]
    df_s1 = df.iloc[s1_list]
    df_s2 = df.iloc[s2_list]
    
    os.makedirs(f'{args.dirname}/{args.dataset}/', exist_ok=True)
    
    print("\nGeneraging negative for training set:")
    df_tr = generate_pair_triplets(args, df_tr, drug_ids)
    filename = f'{args.dirname}/{args.dataset}/pair_pos_neg_triplets_train.csv'
    df_tr.to_csv(filename, index=False)
    print(f'Training set ({len(df_tr)}) saved as {filename}!')
    
    print("\nGeneraging negative for S1 set:")
    df_s1 = generate_pair_triplets(args, df_s1, drug_ids)
    filename = f'{args.dirname}/{args.dataset}/pair_pos_neg_triplets_S1.csv'
    df_s1.to_csv(filename, index=False)
    print(f'S1 set ({len(df_s1)}) saved as {filename}!')
    
    print("\nGeneraging negative for S2 set:")
    df_s2 = generate_pair_triplets(args, df_s2, drug_ids)
    filename = f'{args.dirname}/{args.dataset}/pair_pos_neg_triplets_S2.csv'
    df_s2.to_csv(filename, index=False)
    print(f'S2 set ({len(df_s2)}) saved as {filename}!')
    
def cold_split_ignore_neg(args):
    drug_ids = []
    id2data_dict = torch.load(os.path.join(args.dirname, f'{args.dataset}.pth'))
    drug_ids = list(id2data_dict.keys())
    
    train_ids, test_ids = train_test_split(drug_ids, test_size=args.test_ratio, random_state=args.seed)
    
    df = pd.read_csv(args.dataset_filename, delimiter=args.delimiter)
    print("\nGeneraging negative...")
    # df = generate_pair_triplets(args, df, drug_ids)
    
    train_list, s1_list, s2_list = [], [], []
    for i in tqdm(range(len(df)), desc="Cold splitting..."):
        pair = df.iloc[i]
        id1, id2 = pair[args.c_id1], pair[args.c_id2]
        if id1 in train_ids and id2 in train_ids:
            train_list.append(i)
        elif id1 in test_ids and id2 in test_ids:
            s1_list.append(i)
        else:
            s2_list.append(i)
    
    df_tr = df.iloc[train_list]
    df_s1 = df.iloc[s1_list]
    df_s2 = df.iloc[s2_list]
    
    os.makedirs(f'{args.dirname}/{args.dataset}/', exist_ok=True)
    
    
    
    df_tr, val_df = train_test_split(df_tr, test_size=0.1, random_state=114514)
    filename = f'{args.dirname}/{args.dataset}/ddi_training.csv'
    df_tr.to_csv(filename, index=False)
    print(f'Training set ({len(df_tr)}) saved as {filename}!')
    
    filename = f'{args.dirname}/{args.dataset}/ddi_validation.csv'
    val_df.to_csv(filename, index=False)
    print(f'val set ({len(val_df)}) saved as {filename}!')
    
    filename = f'{args.dirname}/{args.dataset}/ddi_tests1.csv'
    df_s1.to_csv(filename, index=False)
    print(f'S1 set ({len(df_s1)}) saved as {filename}!')
    

    filename = f'{args.dirname}/{args.dataset}/ddi_tests2.csv'
    df_s2.to_csv(filename, index=False)
    print(f'S2 set ({len(df_s2)}) saved as {filename}!')

# =============================== Dataset & DataLoader ===============================

class DDIDataset(Dataset):
    def __init__(self, df, id2data_dict):
        self.df = df
        self.id2data_dict = id2data_dict
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        Drug1_ID, Drug2_ID, Y, Neg_samples = row['Drug1_ID'], row['Drug2_ID'], row['Y'], row['Neg samples']
        
        Neg_ID, Ntype = Neg_samples.split('$')
        h_graph = self.id2data_dict.get(Drug1_ID)
        t_graph = self.id2data_dict.get(Drug2_ID)
        n_graph = self.id2data_dict.get(Neg_ID)

        pos_pair_h = h_graph
        pos_pair_t = t_graph

        if Ntype == 'h':
            neg_pair_h = n_graph
            neg_pair_t = t_graph
        else:
            neg_pair_h = h_graph
            neg_pair_t = n_graph  
            
        return {'pos_pair_h': pos_pair_h, 'pos_pair_t': pos_pair_t,
                'neg_pair_h': neg_pair_h, 'neg_pair_t': neg_pair_t,
                'rel': Y}
    
    def unique_data_list(self):
        print("Get unique data_list...")
        drug_set = set()
        for i, row in self.df.iterrows():
            Drug1_ID, Drug2_ID, _, Neg_samples = row['Drug1_ID'], row['Drug2_ID'], row['Y'], row['Neg samples']
            Neg_ID, Ntype = Neg_samples.split('$')
            drug_set.add(Drug1_ID)
            drug_set.add(Drug2_ID)
            drug_set.add(Neg_ID)
            
        drug_list = []
        for drug_id in drug_set:
            drug_list.append(self.id2data_dict[drug_id])
        
        return drug_list
            
            
class DDIDataLoader(DataLoader):
    def __init__(self, data_list, **kwargs):
        super().__init__(data_list, collate_fn=self.collate_fn, **kwargs)
        
    def collate_fn(self, batch):
        head_list = []
        tail_list = []
        label_list = []
        rel_list = []
        
        for pairs in batch:
            pos_pair_h = pairs['pos_pair_h']
            pos_pair_t = pairs['pos_pair_t']
            neg_pair_h = pairs['neg_pair_h']
            neg_pair_t = pairs['neg_pair_t']
            rel = pairs['rel']
            
            head_list.append(pos_pair_h)
            head_list.append(neg_pair_h)
            tail_list.append(pos_pair_t)
            tail_list.append(neg_pair_t)

            rel_list.append(torch.LongTensor([rel]))
            rel_list.append(torch.LongTensor([rel]))
            
            label_list.append(torch.FloatTensor([1]))
            label_list.append(torch.FloatTensor([0]))
            
        head_pairs = Batch.from_data_list(head_list)
        tail_pairs = Batch.from_data_list(tail_list)
        rel = torch.cat(rel_list, dim=0)
        label = torch.cat(label_list, dim=0)

        return head_pairs, tail_pairs, rel, label


def get_benchmark_loader(dataset='drugbank', cold='S1',
                         batch_size=64, seed=114514, num_workers=0):
    assert cold in ['S1', 'S2']
    train_path = f'dataset/data/DDI/processed/{dataset}/train.csv'
    val_path = f'dataset/data/DDI/processed/{dataset}/val.csv'
    test_path = f'dataset/data/DDI/processed/{dataset}/{cold}.csv'
    id2data_dict = torch.load(f'dataset/data/DDI/processed/{dataset}.pth')
               
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    val_df = pd.read_csv(val_path)
    # train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=114514)
    
    # split_path = f'dataset/data/DDI/processed/{dataset}/{cold}'
    # train_df.to_csv(os.path.join(split_path, 'ddi_train'), index=False)
    
    train_dataset = DDIDataset(train_df, id2data_dict)
    test_dataset = DDIDataset(test_df, id2data_dict)
    val_dataset = DDIDataset(val_df, id2data_dict)
    
    torch.manual_seed(seed=seed)
    
    loader_tr = DDIDataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers)
    loader_te = DDIDataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                               num_workers=num_workers)
    loader_va = DDIDataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                               num_workers=num_workers)
    
    return loader_tr, loader_va, loader_te

    
    #%%
if __name__ == "__main__":
    from databuild import from_smiles
    import argparse
    
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--dataset', type=str, default='drugbank', choices=['drugbank', 'twosides'], 
                            help='Dataset to preprocess.')
    parser.add_argument('-n', '--neg_ent', type=int, default=1, help='Number of negative samples')
    parser.add_argument('-s', '--seed', type=int, default=114514, help='Seed for the random number generator')
    parser.add_argument('-t_r', '--test_ratio', type=float, default=0.2)
    parser.add_argument('-n_f', '--n_folds', type=int, default=3)
    parser.add_argument('-n_p', '--num_processor', type=int, default=8,
                        help='multi processing num processors')
    dataset_columns_map = {
        'drugbank': ('d1', 'd2', 'smiles1', 'smiles2', 'type'),
        'twosides': ('Drug1_ID', 'Drug2_ID', 'Drug1', 'Drug2', 'New Y'),
    }

    dataset_file_name_map = {
        'drugbank': ('data/DDI/drugbank.csv', ','),
        'twosides': ('data/DDI/twosides_ge_500.csv', ',')
    }

    args = parser.parse_args()
    args.dataset = args.dataset.lower()
    args.datamaker = from_smiles
    args.c_id1, args.c_id2, args.c_s1, args.c_s2, args.c_y = dataset_columns_map[args.dataset]
    args.dataset_filename, args.delimiter = dataset_file_name_map[args.dataset]
    args.dirname = 'data/DDI/processed'
    args.random_num_gen = np.random.RandomState(args.seed)
    os.makedirs(args.dirname, exist_ok=True)
#%%
    prepare_drug_data(args)
    # generate_pair_triplets(args)
    #%%
    # cold_split(args)
    cold_split_ignore_neg(args)













































