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


"""
The following funcs are copied and adaptedfrom https://github.com/thinng/GraphDTA
"""
seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
# seq_voc = "ABDEFGHJKLMNPQRSTUVW"
seq_dict = {v:(i+1) for i,v in enumerate(seq_voc)}

def seq_cat(prot, max_seq_len):
    x = torch.zeros(max_seq_len, dtype=torch.int32)
    for i, ch in enumerate(prot[:max_seq_len]): 
        x[i] = seq_dict[ch]
    return x  

CHARPROTSET = {
    "A": 1,
    "C": 2,
    "B": 3,
    "E": 4,
    "D": 5,
    "G": 6,
    "F": 7,
    "I": 8,
    "H": 9,
    "K": 10,
    "M": 11,
    "L": 12,
    "O": 13,
    "N": 14,
    "Q": 15,
    "P": 16,
    "S": 17,
    "R": 18,
    "U": 19,
    "T": 20,
    "W": 21,
    "V": 22,
    "Y": 23,
    "X": 24,
    "Z": 25,
}

def integer_label_protein(sequence, max_length=1200):
    """
    Integer encoding for protein string sequence.
    Args:
        sequence (str): Protein string sequence.
        max_length: Maximum encoding length of input protein string.
    """
    encoding = np.zeros(max_length)
    for idx, letter in enumerate(sequence[:max_length]):
        try:
            letter = letter.upper()
            encoding[idx] = CHARPROTSET[letter]
        except KeyError:
            logging.warning(
                f"character {letter} does not exists in sequence category encoding, skip and treat as " f"padding."
            )
    return encoding

def encode_seq_deepdta(id_seq_dict, max_seq_len=1200):
    target_embd_dict = {}
    for k, v in tqdm(id_seq_dict.items(), desc='Init protein embeds...'):
        seq = seq_cat(v, max_seq_len)
        target_embd_dict[k] = seq
        
    return target_embd_dict

def read_pickle(filename):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    print(filename, " successfully loaded")
    return obj

def df_reg2cls(df, threshold=6):
    # chembl ratio 0.772
    df.loc[df.ec50 < threshold, 'ec50']=0
    df.loc[df.ec50 >= threshold, 'ec50']=1
    df['ec50'] = df['ec50'].apply(int)
    return df
    



class DTIDataLoader(DataLoader):
    def __init__(self, data_list, **kwargs):
        super().__init__(data_list, collate_fn=self.collate_fn, **kwargs)
        
    def collate_fn(self, batch):
        mols = []
        tars = []
        ys = []
        for mol, tar, y in batch:
            mols.append(mol)
            tars.append(tar)
            ys.append(y)
        mols = Batch.from_data_list(mols)
        tars = torch.stack(tars, dim=0)
        ys = torch.tensor(ys, dtype=torch.float32)
        return (mols, tars, ys)
    
def get_loader_random(dataset, batch_size=64, seed=114514, num_workers=0):
    data_tr, data_te = train_test_split(dataset, test_size=0.2, random_state=seed)
    data_te, data_va = train_test_split(data_te, test_size=0.5, random_state=seed)
    torch.manual_seed(seed=seed)
    loader_tr = DTIDataLoader(data_tr, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers)
    loader_va = DTIDataLoader(data_va, batch_size=batch_size, shuffle=False,
                               num_workers=num_workers)
    loader_te = DTIDataLoader(data_te, batch_size=batch_size, shuffle=False,
                               num_workers=num_workers)
    return loader_tr, loader_va, loader_te

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
######################
# from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer
# from torch_geometric.utils import from_dgl, sort_edge_index
# class easy_data:
#     def __init__(self, datamaker, get_fp=False, with_hydrogen=False, with_coordinate=False, node_featurizer=CanonicalAtomFeaturizer(), edge_featurizer=CanonicalBondFeaturizer(self_loop=False),
#                   seed=123):
#         self.datamaker = datamaker
#         self.node_featurizer = node_featurizer
#         self.edge_featurizer = edge_featurizer
#         self.get_fp = get_fp
#         self.with_hydrogen = with_hydrogen
#         self.with_coordinate = with_coordinate
#         self.seed = seed
#         self.rdkit_fp = False
#         self.mask_attr = False

#     def process(self, smi, mol):
#         g = smiles_to_bigraph(smi, add_self_loop=False, node_featurizer=self.node_featurizer, edge_featurizer=self.edge_featurizer)
#         g.ndata['x'] = g.ndata.pop('h')
#         g.edata['edge_attr'] = g.edata.pop('e')
#         data = from_dgl(g)
#         data.edge_index, data.edge_attr = sort_edge_index(data.edge_index.to(torch.long), data.edge_attr.to(torch.float32))
#         data.x = data.x.to(torch.float32)
#         data.smiles = smi
#         return data
    
#     def __repr__(self):
#         info = f"easy_data(rdkit_fp={self.rdkit_fp}, fpSize={self.fpSize}, mask_attr={self.mask_attr})"
#         return info
######################

def prepare_dataset(datamaker, num_processor, dataset='bindingdb'):
    root = f'./data/DTI/{dataset}'
    full = pd.read_csv(os.path.join(root, 'full.csv'))
    smiles = set(full['SMILES'].tolist())
    protein = set(full['Protein'].tolist())
    mols = [Chem.MolFromSmiles(s) for s in tqdm(smiles, desc='smiles2mol...')]
    ed = easy_data(datamaker, get_fp=False, with_hydrogen=False, with_coordinate=False, seed=123)
    data_list = list(Parallel(n_jobs=num_processor)(delayed(ed.process)(smi, mol)
                                         for smi, mol in tqdm(zip(smiles, mols),
                                                                     total=len(smiles),
                                                                     desc='Generating drug data...')))
    drug_dict = {}
    for data in data_list: drug_dict[data.smiles] = data
    
    prot_dict = {}
    for seq in tqdm(protein, desc='seq2embd...'):
        prot_dict[seq] = torch.tensor(integer_label_protein(seq, 1200), dtype=torch.float32)
        
    torch.save(drug_dict, os.path.join(root, 'drugs.pth'))
    torch.save(prot_dict, os.path.join(root, 'protein.pth'))


class BenchmarkDataset(Dataset):
    def __init__(self, df_path, drug_dict, prot_dict):
        self.df = pd.read_csv(df_path)        
        self.drug_dict = drug_dict
        self.prot_dict = prot_dict
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        drug = self.drug_dict[row['SMILES']]
        prot = self.prot_dict[row['Protein']]
        y = row['Y']
        return (drug, prot, y)
    
    
def get_benchmark_loader(task='bindingdb', split='random',
                         batch_size=64, seed=114514, num_workers=0):
    root = f'dataset/data/DTI/{task}/{split}'
    drug_dict = torch.load(f'dataset/data/DTI/{task}/drugs.pth')
    prot_dict = torch.load(f'dataset/data/DTI/{task}/protein.pth')                     
    train_dataset = BenchmarkDataset(os.path.join(root, 'train.csv'), drug_dict, prot_dict)
    test_dataset = BenchmarkDataset(os.path.join(root, 'test.csv'), drug_dict, prot_dict)
    val_dataset = BenchmarkDataset(os.path.join(root, 'val.csv'), drug_dict, prot_dict)
    torch.manual_seed(seed=seed)
    loader_tr = DTIDataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, drop_last=True)
    loader_te = DTIDataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                               num_workers=num_workers, drop_last=False)
    loader_va = DTIDataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                               num_workers=num_workers, drop_last=False)
    return loader_tr, loader_va, loader_te



# ********** ALDH2 筛选 *****************************************************************************
class ScreenDataset(Dataset):
    def __init__(self, df, mol_dict, tar_dict):
        self.df = df
        self.data_list = []
        for idx, row in df.iterrows():
            drug = mol_dict[row['Smiles']]
            target = tar_dict[row['Target ChEMBL ID']]
            y = torch.tensor([row['class']], dtype=torch.float32)
            self.data_list.append((drug, target, y))
            if row['Target ChEMBL ID'] == 'CHEMBL1935':
                self.data_list += [(drug, target, y)] * 10
                if y == 1:
                    self.data_list += [(drug, target, y)] * 100
       
        
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        return self.data_list[index]
    
class ScreenLoader(DataLoader):
    def __init__(self, data_list, **kwargs):
        super().__init__(data_list, collate_fn=self.collate_fn, **kwargs)
        
    def collate_fn(self, batch):
        drugs = []
        tars = []
        ys = []
        for drug, target, y in batch:
            drugs.append(drug)
            tars.append(target)
            ys.append(y)
        drugs = Batch.from_data_list(drugs)
        tars = torch.stack(tars, dim=0)
        ys = torch.cat(ys, dim=0)
        return (drugs, tars, ys)

def get_screen_loader(batch_size=64, seed=114514, num_workers=0):
    df = pd.read_csv('dataset/data/DTI/ChemblICEC.csv')
    df = df.dropna()
    print("Loading drugs...")
    drug_dict = torch.load('dataset/data/DTI/screen/drugs.pth')
    print("Loading proteins...")
    prot_dict = torch.load('dataset/data/DTI/screen/protein.pth')
    print("Generating dataset...")
    dataset = ScreenDataset(df, drug_dict, prot_dict)
    train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=seed)
    val_dataset, test_dataset = train_test_split(test_dataset, test_size=0.5, random_state=seed)
    loader_tr = ScreenLoader(train_dataset, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, drop_last=True)
    loader_te = ScreenLoader(test_dataset, batch_size=batch_size, shuffle=False,
                               num_workers=num_workers, drop_last=False)
    loader_va = ScreenLoader(val_dataset, batch_size=batch_size, shuffle=False,
                               num_workers=num_workers, drop_last=False)
    return loader_tr, loader_va, loader_te

    
def prepare_screen_dataset(datamaker, num_processor):
    path = './data/DTI/ChemblICEC.csv'
    full = pd.read_csv(path)
    full = full.dropna()
    smiles = set(full['Smiles'].tolist())
    # protein = set(full['Target ChEMBL ID'].tolist())
    mols = [Chem.MolFromSmiles(s) for s in tqdm(smiles, desc='smiles2mol...')]
    ed = easy_data(datamaker, get_fp=False, with_hydrogen=False, with_coordinate=False, seed=123)
    data_list = list(Parallel(n_jobs=num_processor)(delayed(ed.process)(smi, mol)
                                         for smi, mol in tqdm(zip(smiles, mols),
                                                                     total=len(smiles),
                                                                     desc='Generating drug data...')))
    drug_dict = {}
    for data in data_list: drug_dict[data.smiles] = data
    
    with open ('./data/DTI/SequenceDict.pkl', 'rb') as f:
        prot_id2seq = pickle.load(f)
        
    prot_dict = {}
    for chembl_id, seq in tqdm(prot_id2seq.items(), desc='seq2embd...'):
        prot_dict[chembl_id] = torch.tensor(integer_label_protein(seq, 1200), dtype=torch.float32)
        
    torch.save(drug_dict, './data/DTI/screen/drugs.pth')
    torch.save(prot_dict, './data/DTI/screen/protein.pth')

class EC50ScreenLoader(DataLoader):
    def __init__(self, data_list, **kwargs):
        super().__init__(data_list, collate_fn=self.collate_fn, **kwargs)
        
    def collate_fn(self, batch):
        mols = []
        for mol in batch:
            mols.append(mol)
        mols = Batch.from_data_list(mols)
        return mols
# **************************************************************************************************

if __name__ == "__main__":
    
    # ==================== Benchmark ====================
    import argparse
    from databuild import from_smiles
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='human', choices=['bindingdb', 'human', 'biosnap'], 
                            help='Dataset to preprocess.')
    parser.add_argument('-n_p', '--num_processor', type=int, default=8,
                        help='multi processing num processors')
    args = parser.parse_args()
    
    prepare_dataset(from_smiles, args.num_processor, args.dataset)
    
    # ===================================================
    
    # prepare_screen_dataset(from_smiles, args.num_processor)
    
    
    
    
    
    
    
    
    
    
    # from databuild import from_smiles
    # root = './data/DTI'
    
    # # for dataset in ['ec50s', 'chembl']: 
    # #     # processing mol
    # #     mol_path = os.path.join(root, f'{dataset}_cid2smi.pkl')
    # #     with open(mol_path, 'rb') as f:
    # #         cid2smi = pickle.load(f)
        
    # #     mol_data_dict = {}
    # #     for cid, smiles in tqdm(cid2smi.items(), total=len(cid2smi)):
    # #         mol = Chem.MolFromSmiles(smiles)
    # #         if mol is not None: 
    # #             data = from_smiles(smiles, mol, get_fp=True, with_coordinate=False)
    # #             mol_data_dict[cid] = data
        
    # #     torch.save(mol_data_dict, os.path.join(root, f'{dataset}_mol_dict.pth'))
    
    # test_df = pd.read_csv(os.path.join(root, 'zinc_test.csv'))
    # smiles_list = test_df['smiles'].tolist()
    # data_list = []
    # for smiles in tqdm(smiles_list):
    #     mol = Chem.MolFromSmiles(smiles)
    #     if mol is not None: 
    #         data = from_smiles(smiles, mol, get_fp=True, with_coordinate=False)
    #         data_list.append(data)

    # torch.save(data_list, os.path.join(root, 'zinc_test_data_list.pth'))
























































