# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 19:07:25 2025

@author: Fanding Xu
"""

import os
import argparse
import torch
import numpy as np
import yaml
from models.model_dti import UnetDTI
from rdkit.Chem.Draw import IPythonConsole
from dataset.databuild import from_smiles

from dataset.databuild_dti import easy_data, integer_label_protein
from utils import mol_with_atom_index, comps_visualize_multi, comps_visualize_single, visual_sep_subs
from rdkit import Chem
from IPython.display import display, SVG, Image
from torch_geometric.data import Batch
from torch_geometric.utils import index_to_mask

import matplotlib.pyplot as plt
from rdkit.Chem import Draw
import matplotlib.offsetbox as offsetbox

parser = argparse.ArgumentParser()

# Add argument
parser.add_argument('--sequence', type=str, required=True,
                    help='protein sequence, required')
parser.add_argument('--smiles', type=str, required=True,
                    help='molecular smiles, required')
parser.add_argument('--name', type=str, required=True,
                    help='name, required')

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

save_root = "visual_demo/DTI"
os.makedirs(save_root, exist_ok=True)


config = yaml.load(open('./config.yaml', 'r'), Loader=yaml.CLoader)  
config['model']['hidden_dim'] = 128
config['deg'] = torch.tensor([   68, 23804, 49733, 32711,  2724,     3,     3])

model = UnetDTI(config).to(device)
model.load_state_dict(torch.load('checkpoint/DTI/bindingdb.pt'))

name = args.name

t = args.sequence
d = args.smiles

mol = Chem.MolFromSmiles(d)
ed = easy_data(from_smiles, get_fp=False, with_hydrogen=False, with_coordinate=False, seed=123)
data_d = ed.process(d, mol)
data_d = Batch.from_data_list([data_d]).to(device)
data_t = torch.tensor(integer_label_protein(t, 1200), dtype=torch.float32).unsqueeze(0).to(device)

model.eval()
with torch.no_grad():
    (x, attn), pred = model(data_d, data_t)
    pred_txt = f"{pred.sigmoid().detach().item():.4f}"
    print(pred.sigmoid().detach().item())
    
    
pool_info = model.pool_info
comps = pool_info['comp']
tars = pool_info['edge_target']
mol = Chem.MolFromSmiles(d)
num_atom = mol.GetNumAtoms()
mol = mol_with_atom_index(Chem.MolFromSmiles(d))
imgs, _, all_sub_atom, all_sub_bond = comps_visualize_multi(mol, comps, tars, data_d.edge_index, only_mol_img=False)

def find_func_atoms(all_sub_atom, num_atom):
    full_set = set(range(num_atom))
    existing_numbers = set(num for sublist in all_sub_atom for num in sublist)
    missing_numbers = full_set - existing_numbers
    func_atoms = [[num] for num in sorted(missing_numbers)]
    return func_atoms
func_atoms = find_func_atoms(all_sub_atom, num_atom)

for png in imgs:
    display(Image(png))

svg = comps_visualize_single(mol, comps, tars, data_d.edge_index, form='svg')
with open(f"{save_root}/single_{name}.svg", 'w') as f:
    f.write(svg)
display(SVG(svg))
#%%  
all_sub_atom = all_sub_atom + func_atoms
all_sub_bond = all_sub_bond + [[] for i in range(len(func_atoms))]
# structures.append(list(range(num_atom)))
sub_attr = []
for sub in all_sub_atom:
    sub_mask = index_to_mask(torch.tensor(sub), num_atom).logical_not()
    with torch.no_grad():
        _, sub_pred = model(data_d, data_t, sub_mask)
        
        # attribution = torch.tanh(pred - sub_pred)
        # attribution = pred.sigmoid() - sub_pred.sigmoid()
        attribution = pred - sub_pred
        # print(sub_pred)
        sub_attr.append(attribution)

attribution_str = []
for tensor in sub_attr:
    values = tensor.cpu().numpy().flatten()  # Convert to CPU, numpy, and flatten the array
    formatted_str = ', '.join([f"{value:.3f}" for value in values])
    attribution_str.append(formatted_str)


img, smiles_list, frgs = visual_sep_subs(mol, all_sub_atom, all_sub_bond,
                                         attribution=attribution_str, show_smiles=False)

if img is not None:
    display(img)

with open(f"{save_root}/sep_{name}.svg", 'w') as f:
    f.write(img.data)

#%%

sub_attr_np = torch.cat(sub_attr, dim=0).cpu().numpy()
attribution = sub_attr_np[:, 0]


attribution = np.tanh(attribution)


def get_mol_img(mol, size=(400, 400)):
    img = Draw.MolToImage(mol, size=size, kekulize=True, options=Draw.DrawingOptions())
    img = img.convert("RGBA")
    datas = img.getdata()
    new_data = []
    for item in datas:
        if item[:3] == (255, 255, 255):  # 纯白色背景部分变为透明
            new_data.append((255, 255, 255, 0))
        else:
            new_data.append(item)
    img.putdata(new_data)  # 增大分子图像尺寸
    return img


indices = np.arange(len(attribution))
plt.rc('font', size=22)
# 创建图和轴
fig, ax = plt.subplots(figsize=(12, 8))

# 添加网格
ax.grid(True, linestyle='--', alpha=0.6, zorder=-1)

# 绘制水平条形图，使用低饱和度的淡粉色和淡蓝色
colors = ['#e6a9a9' if value < 0 else '#9fc5e8' for value in attribution]
ax.barh(indices, attribution, color=colors, edgecolor='lightgrey', linewidth=1, zorder=2)

# 设置x轴的零点在中间
ax.axvline(0, color='black', lw=1)

# 设置标签
ax.set_yticks([])
ax.set_yticklabels([])
ax.set_yticks([])
ax.set_ylabel('')
ax.set_xlabel('Attribution Value')

# ax.set_yticks(indices)
# ax.set_yticklabels(smiles_list)
# ax.set_xlabel('Attribution Value')

# 计算x轴的范围并调整
zoom = 0.4

x_min = min(attribution) - zoom
x_max = max(attribution) + zoom
x_min, x_max = -1.25, 1.25
# ax.set_xlim(-1.2, 1.2)
ax.set_xlim(x_min, x_max)

# 设置左侧和右侧的底色
ax.axvspan(x_min, 0, facecolor='#fde0dc', alpha=0.3, zorder=-2)  # 左边淡红色
ax.axvspan(0, x_max, facecolor='#d0e7ff', alpha=0.3, zorder=-2)  # 右边淡蓝色

# 添加分子图像
for i, (smiles, sub_mol) in enumerate(zip(smiles_list, frgs)):
    # mol = Chem.MolFromSmiles(smiles)
    if sub_mol:
        # img = Draw.MolToImage(sub_mol, size=(100, 100), fitImage=False)  # 增大分子图像尺寸
        img = get_mol_img(sub_mol, size=(200, 200))
        
        imagebox = offsetbox.OffsetImage(img, zoom=0.5)  # 增大图像缩放比例
        bias = 0.2
        if attribution[i] < 0: bias *= -1
        ab = offsetbox.AnnotationBbox(imagebox, (attribution[i]+bias, i), frameon=False, box_alignment=(0.5, 0.5))
        
        ax.add_artist(ab)
        ab.set_zorder(4)

# 显示图
plt.tight_layout()

plt.savefig(f"{save_root}/{name}.svg", format='svg')
plt.show()


























































