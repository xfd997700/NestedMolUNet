# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 16:56:49 2023

@author: Fanding Xu
"""

import pandas as pd
import numpy as np
import torch
import os
from joblib import Parallel, delayed
import argparse
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import CanonSmiles
if __name__ == "__main__":
    from databuild import from_smiles, read_pickle
    from utils import MaskAtom, MaskAtomMotif
    from databuild_pretrain import make_graph_data, easy_data
else:
    from .databuild import from_smiles, read_pickle
    from .utils import MaskAtom, MaskAtomMotif
    from .databuild_pretrain import make_graph_data, easy_data


def mol_filter_by_smi(smiles, labels=None):
    new_smiles, mols = [], []
    if labels is not None:
        assert len(labels) == len(smiles)
        new_labels = []
        
    for i, smi in tqdm(enumerate(smiles), total=len(smiles)):
        # standardize smiles
        smi = CanonSmiles(smi)
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None and mol.GetNumAtoms() > 1:
                mols.append(mol)
                new_smiles.append(smi)
                if labels is not None:
                    new_labels.append(labels[i])
        except:
            continue
    if labels is not None:
        return mols, new_smiles, new_labels
    return mols, new_smiles

def generate_admet_data(args, root='data/admet_open_data/admet_labelled_data'):
    for name in os.listdir(root):
        data = pd.read_csv(os.path.join(root, f'{name}/data.csv'))
        if isinstance(data['label'].iloc[0], np.float64):
            task_type = 'reg'
        else: task_type = 'cls'
        smiles = data['SMILES'].tolist()
        labels = data['label'].tolist()
        print(f'smiles2mol for {name}...')
        mols, smiles, labels = mol_filter_by_smi(smiles, labels)
        ed = easy_data(get_desc=True, get_fp=False, get_comp=False,
                       with_hydrogen=False, with_coordinate=False, seed=args.seed)
        ed.int_rdkit_fpgen(args)
        data_list = [ed.process(smi, mol, label)
                     for smi, mol, label in tqdm(zip(smiles, mols, labels),
                                                 total=len(smiles),
                                                 desc='Generating data...')]
        print(f'Saving datalist: {name}')
        torch.save(data_list, f'data/admet/processed/{name}_{task_type}.pth')
    return data_list


def generate_admet_data_mp(args, root='data/admet_open_data/admet_labelled_data'):
    smiles_list = []
    for name in os.listdir(root):
        data = pd.read_csv(os.path.join(root, f'{name}/data.csv'))
        smiles = data['SMILES'].tolist()
        smiles_list += smiles
    
    print('smiles2mol for admet...')
    smiles_set = set(smiles_list)
    mols, smiles = mol_filter_by_smi(smiles_set)
    ed = easy_data(get_desc=True, get_fp=False, get_comp=False,
                   with_hydrogen=False, with_coordinate=False, seed=args.seed)
    ed.int_rdkit_fpgen(args)
    data_list = list(Parallel(n_jobs=args.num_processor)(delayed(ed.process)(smi, mol)
                                         for smi, mol in tqdm(zip(smiles, mols),
                                                              total=len(smiles),
                                                              desc='Generating data...')))
    data_dict = {}
    for data in data_list:
        data_dict[data.smiles] = data
    print('Saving admet smi2data dict')
    torch.save(data_dict, 'data/admet/processed/admet.pth')


def generate_hiv_data_mp(args, path='data/admet/HIV.csv'):
    df = pd.read_csv(path)
    df = df.fillna(-1)
    smiles = df['smiles'].tolist()
    labels = df['HIV_active'].tolist()

    print('smiles2mol for HIV...')
    mols, smiles, labels = mol_filter_by_smi(smiles, labels)
    
    ed = easy_data(get_desc=True, get_fp=False, get_comp=False,
                   with_hydrogen=False, with_coordinate=False, seed=args.seed)
    ed.int_rdkit_fpgen(args)
    data_list = list(Parallel(n_jobs=args.num_processor)(delayed(ed.process)(smi, mol, label)
                                         for smi, mol, label in tqdm(zip(smiles, mols, labels),
                                                                     total=len(smiles),
                                                                     desc='Generating data...')))
    print('Saving hiv data')
    torch.save(data_list, 'data/admet/processed/hiv.pth')

def generate_pcba_data_mp(args, path='data/admet/PCBA.csv'):
    df = pd.read_csv(path)
    df = df.fillna(-1)
    smiles = df['smiles'].tolist()
    labels = df[df.columns[:128]]
    labels = labels.fillna(-1)
    labels = labels.values.tolist()
    print('smiles2mol for PCBA...')
    mols, smiles, labels = mol_filter_by_smi(smiles, labels)
    
    ed = easy_data(get_desc=True, get_fp=False, get_comp=False,
                   with_hydrogen=False, with_coordinate=False, seed=args.seed)
    ed.int_rdkit_fpgen(args)
    data_list = list(Parallel(n_jobs=args.num_processor)(delayed(ed.process)(smi, mol, label)
                                         for smi, mol, label in tqdm(zip(smiles, mols, labels),
                                                                     total=len(smiles),
                                                                     desc='Generating data...')))
    print('Saving PCBA data')
    torch.save(data_list, 'data/admet/processed/pcba.pth')
    # return data_list

#%%
if __name__ == "__main__":
    from rdkit import RDLogger  
    RDLogger.DisableLog('rdApp.*')  

    parser = argparse.ArgumentParser()
    parser.add_argument('--fpSize', type=int, default=1024,
                        help='size of rdkit fingerprint for graph similarity loss {default: 1024}')
    parser.add_argument('--num_processor', type=float, default=8,
                        help='multi process num processors (default: 8)')
    parser.add_argument('--seed', type=float, default=123,
                        help='Random seed(default: 123)')
    args = parser.parse_args()
    
    generate_admet_data_mp(args)
    generate_pcba_data_mp(args)
    generate_hiv_data_mp(args)











































