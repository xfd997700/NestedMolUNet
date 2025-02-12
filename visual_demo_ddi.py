# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 21:04:54 2024

@author: Fanding Xu
"""

import os
import torch
import numpy as np
import yaml
from models.model_ddi import UnetDDI_SSI
from rdkit.Chem.Draw import IPythonConsole
from dataset.databuild import from_smiles
import argparse
from dataset.databuild_ddi import easy_data
from utils import mol_with_atom_index, comps_visualize_multi, visual_sep_subs
from rdkit import Chem
from IPython.display import display, SVG
from torch_geometric.data import Batch

import matplotlib.pyplot as plt
from rdkit.Chem import Draw
import matplotlib.offsetbox as offsetbox


from torch_geometric.utils import index_to_mask
import copy


parser = argparse.ArgumentParser()

# Add argument
parser.add_argument('--h_smiles', type=str, required=True,
                    help='head molecular smiles, required')
parser.add_argument('--h_name', type=str, required=True,
                    help='head molecular name, required')
parser.add_argument('--t_smiles', type=str, required=True,
                    help='tail molecular smiles, required')
parser.add_argument('--t_name', type=str, required=True,
                    help='tail molecular name, required')

args = parser.parse_args()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

save_root = "visual_demo/DDI"
os.makedirs(save_root, exist_ok=True)

h = args.h_smiles
t = args.t_smiles

config = yaml.load(open('./config.yaml', 'r'), Loader=yaml.CLoader)  
config['model']['num_pool_layer'] = 4
config['model']['hidden_dim'] = 64
config['deg'] = torch.tensor([  123,  8162, 16034, 11158,   999,     0,     2])
model = UnetDDI_SSI(config).to(device)
model.load_state_dict(torch.load('checkpoint/DDI/drugbank.pt'))


mol_h = Chem.MolFromSmiles(h)
# Chem.RemoveStereochemistry(mol_h)
mol_t = Chem.MolFromSmiles(t)
ed = easy_data(from_smiles, get_fp=True, with_hydrogen=False, with_coordinate=False, seed=123)
data_h = ed.process(h, mol_h)
data_t = ed.process(t, mol_t)
data_h = Batch.from_data_list([data_h]).to(device)
data_t = Batch.from_data_list([data_t]).to(device)
rel = torch.LongTensor([84]).to(device)
label = torch.LongTensor([1]).to(device)

tri = (data_h, data_t, rel, label)

model.eval()
with torch.no_grad():
    pred, _ = model(tri)
    pred_txt = f"{pred.sigmoid().detach().item():.4f}-{args.h_name}-{args.t_name}"
    print(pred.sigmoid().detach().item())
    h_info, t_info = model.unet.pool_info


flat_index = torch.argmax(model.a)
row, col = divmod(flat_index.item(), model.a.size(1))


comps_h = h_info['comp']
tars_h = h_info['edge_target']
imgs_h, _, all_sub_atom_h, all_sub_bond_h = comps_visualize_multi(mol_h, comps_h, tars_h,
                                                                  data_h.edge_index, only_mol_img=False, 
                                                                  form='svg', sub_layer=row)
img_h = imgs_h[row]
display(SVG(img_h))
with open(f"{save_root}/ddi_h_{pred_txt}.svg", 'w') as f:
    f.write(img_h)
    
comps_t = t_info['comp']
tars_t = t_info['edge_target']
imgs_t, _, all_sub_atom_t, all_sub_bond_t= comps_visualize_multi(mol_t, comps_t, tars_t,
                                                                 data_t.edge_index, only_mol_img=False, 
                                                                 form='svg', sub_layer=col)
img_t = imgs_t[col]
display(SVG(img_t))
with open(f"{save_root}/ddi_t_{pred_txt}.svg", 'w') as f:
    f.write(img_t)


def find_func_atoms(all_sub_atom, num_atom):
    full_set = set(range(num_atom))
    existing_numbers = set(num for sublist in all_sub_atom for num in sublist)
    missing_numbers = full_set - existing_numbers
    func_atoms = [[num] for num in sorted(missing_numbers)]
    return func_atoms

num_atom_h = mol_h.GetNumAtoms()
func_atoms_h = find_func_atoms(all_sub_atom_h, num_atom_h)
num_atom_t = mol_t.GetNumAtoms()
func_atoms_t = find_func_atoms(all_sub_atom_t, num_atom_t)

all_sub_atom_h = all_sub_atom_h + func_atoms_h
all_sub_bond_h = all_sub_bond_h + [[] for i in range(len(func_atoms_h))]

all_sub_atom_t = all_sub_atom_t + func_atoms_t
all_sub_bond_t = all_sub_bond_t + [[] for i in range(len(func_atoms_t))]

#%%

model.eval()
with torch.no_grad():
    xs_h, xs_t = model.unet(data_h, data_t)

def get_masked_repr(x, sub, num_atom):
    sub_mask = index_to_mask(torch.tensor(sub), num_atom).logical_not()
    return x[sub_mask]

sub_attr_h = []
for sub in all_sub_atom_h:
    xs_h_m = copy.deepcopy(xs_h)
    xs_h_m[row] = get_masked_repr(xs_h_m[row], sub, num_atom_h)
    model.eval()
    with torch.no_grad():
        sub_pred = model.mask_pred(xs_h_m, xs_t, rel)
        attribution = pred - sub_pred
        sub_attr_h.append(attribution)

sub_attr_t = []
for sub in all_sub_atom_t:
    xs_t_m = copy.deepcopy(xs_t)
    xs_t_m[col] = get_masked_repr(xs_t_m[col], sub, num_atom_t)
    model.eval()
    with torch.no_grad():
        sub_pred = model.mask_pred(xs_h, xs_t_m, rel)
        attribution = pred - sub_pred
        sub_attr_t.append(attribution)



attribution_str_h = []
for tensor in sub_attr_h:
    values = tensor.cpu().numpy().flatten()  # Convert to CPU, numpy, and flatten the array
    formatted_str = ', '.join([f"{value:.3f}" for value in values])
    attribution_str_h.append(formatted_str)


img, smiles_list, frgs_h = visual_sep_subs(mol_h, all_sub_atom_h, all_sub_bond_h,
                                         attribution=attribution_str_h, show_smiles=False)

if img is not None:
    display(img)
with open(f"{save_root}/sep_h_{pred_txt}.svg", 'w') as f:
    f.write(img.data)


attribution_str_t = []
for tensor in sub_attr_t:
    values = tensor.cpu().numpy().flatten()  # Convert to CPU, numpy, and flatten the array
    formatted_str = ', '.join([f"{value:.3f}" for value in values])
    attribution_str_t.append(formatted_str)


img, smiles_list, frgs_t = visual_sep_subs(mol_t, all_sub_atom_t, all_sub_bond_t,
                                         attribution=attribution_str_t, show_smiles=False)

if img is not None:
    display(img)
with open(f"{save_root}/sep_t_{pred_txt}.svg", 'w') as f:
    f.write(img.data)
#%%



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



def get_plot(sub_attr, frgs, save_path, with_mol=False):
    sub_attr_np = torch.cat(sub_attr, dim=0).cpu().numpy()
    attribution = sub_attr_np
    attribution = np.tanh(attribution)
    
    
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
    if with_mol:
        for i, sub_mol in enumerate(frgs):
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
    plt.savefig(save_path, format='svg')
    plt.show()



save_path = f"{save_root}/plot_h_{pred_txt}.svg"
get_plot(sub_attr_h, frgs_h, save_path, with_mol=False)
save_path = f"{save_root}/plot_t_{pred_txt}.svg"
get_plot(sub_attr_t, frgs_h, save_path, with_mol=False)















