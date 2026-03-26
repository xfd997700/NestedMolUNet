#!/usr/bin/env python3
"""一键化的 DTI 筛选脚本：给定 Uniprot ID，提取 unibiomap 数据，预测并计算 QED/Lipinski。

输出（放在 results/{uniprot_id} 下）:
- unibiomap_dti.csv
- prediction_result.csv

用法示例:
	python screen_dti_by_uniprot.py --uniprot P05091 --dataset custom --checkpoint checkpoint/DTI/custom.pt
"""
import os
import argparse
import pickle
import yaml
import torch
import pandas as pd
from tqdm import tqdm

from screen_utils import (
	extract_unibiomap_for_uniprots,
	prepare_screening_data, 
	filter_and_compute_properties,
	find_common_molecules
)
from dataset.databuild_dti import DTIDataLoader
from models.model_dti import UnetDTI
from utils import set_seed, get_deg_from_list


class ScreeningDataset(object):
	def __init__(self, df, drug_dict, prot_dict):
		self.df = df
		self.drug_dict = drug_dict
		self.prot_dict = prot_dict

	def __len__(self):
		return len(self.df)

	def __getitem__(self, index):
		row = self.df.iloc[index]
		g = self.drug_dict[row['SMILES']]
		t = self.prot_dict[row['Protein']]
		y = row['Y'] if 'Y' in row else -1.0
		return (g, t, y)


def predict(df, model, device, drug_dict, prot_dict, batch_size=64):
	dataset = ScreeningDataset(df, drug_dict, prot_dict)
	loader = DTIDataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

	preds = []
	model.eval()
	with torch.no_grad():
		for g, t, y in tqdm(loader, desc='Predicting'):
			g = g.to(device)
			t = t.to(device)
			_, pred_logits = model(g, t)
			pred_probs = pred_logits.sigmoid().view(-1).cpu().numpy()
			preds.extend(pred_probs.tolist())

	return preds


def resolve_uniprot_inputs(uniprot_args):
	"""支持直接传 Uniprot ID，或传入包含 ID 列表的 pkl 文件。"""
	resolved = []
	for item in uniprot_args:
		if isinstance(item, str) and item.lower().endswith('.pkl') and os.path.isfile(item):
			with open(item, 'rb') as f:
				loaded = pickle.load(f)

			if isinstance(loaded, pd.Series):
				loaded_ids = loaded.dropna().astype(str).tolist()
			elif isinstance(loaded, (list, tuple, set)):
				loaded_ids = [str(x) for x in loaded if x is not None and str(x).strip()]
			else:
				raise ValueError(f'PKL 文件内容必须是 list/tuple/set/pd.Series，当前类型: {type(loaded)}')

			resolved.extend(loaded_ids)
			print(f'Loaded {len(loaded_ids)} uniprot ids from pkl: {item}')
		else:
			resolved.append(item)

	# 去重并保持顺序
	unique_ids = list(dict.fromkeys([str(x).strip() for x in resolved if str(x).strip()]))
	if not unique_ids:
		raise ValueError('未解析到有效的 Uniprot ID，请检查 --uniprot 输入。')

	return unique_ids


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--uniprot', nargs='+', required=True, help='Uniprot ID(s) 或包含 ID 列表的 pkl 文件路径')
	parser.add_argument('--dataset', default='glass', help='训练时使用的数据集名（用于 checkpoint）')
	parser.add_argument('--checkpoint', default=None, help='模型权重路径，若未提供将按 dataset 找')
	parser.add_argument('--config', default='config.yaml', help='配置文件路径')
	parser.add_argument('--device', default='cuda:0', help='设备，例如 cuda:0 或 cpu')
	parser.add_argument('--batch-size', type=int, default=64)
	parser.add_argument('--num-processors', type=int, default=8)
	parser.add_argument('--seed', type=int, default=42)
	parser.add_argument('--results-root', default='results')
	parser.add_argument('--find-common', action='store_true', help='是否寻找共同分子并保存结果')
	parser.add_argument('--merge-pred', action='store_true', help='合并预测模式：将所有 uniprot 提取结果合并后统一预测')
	parser.add_argument('--conf-thres', type=float, default=0.9, help='提取时保留 conf >= 阈值的记录，默认 0.9')
	args = parser.parse_args()

	set_seed(args.seed)

	uniprot_list = resolve_uniprot_inputs(args.uniprot)
	results_root = args.results_root
	
	results_map = {} # Store dataframe for each uniprot

	cfg = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
	cfg['model']['hidden_dim'] = cfg['model'].get('hidden_dim', 128)
	
	device = torch.device(args.device if torch.cuda.is_available() and 'cuda' in args.device else 'cpu')
	
	if args.checkpoint:
		ckpt_path = args.checkpoint
	else:
		ckpt_path = f'checkpoint/DTI/{args.dataset}.pt'

	if not os.path.exists(ckpt_path):
		raise FileNotFoundError(f'未找到权重: {ckpt_path}')
	
	state_dict = torch.load(ckpt_path, map_location=device)

	if args.merge_pred:
		print(f"\nRunning merged prediction for {len(uniprot_list)} proteins...")
		merge_out_dir = results_root
		os.makedirs(merge_out_dir, exist_ok=True)

		print('批量提取 unibiomap 条目...')
		df = extract_unibiomap_for_uniprots(
			uniprot_ids=uniprot_list,
			save_dir=merge_out_dir,
			save_csv_name='unibiomap_dti.csv',
			write_fasta=False,
			show_progress=True,
			conf_thres=args.conf_thres,
		)

		if df.empty:
			raise RuntimeError('合并模式下未提取到任何条目。')

		df = df.drop_duplicates(subset=['UCI', 'SMILES', 'Protein'])
		print(f'Original merged count: {len(df)}')

		print('计算属性并过滤无效分子...')
		df = filter_and_compute_properties(df)
		if df.empty:
			raise RuntimeError('合并模式下过滤后无有效分子。')

		print('预处理分子与蛋白...')
		drug_dict, prot_dict = prepare_screening_data(df, num_processors=args.num_processors)

		cfg['deg'] = get_deg_from_list(list(drug_dict.values()))
		model = UnetDTI(cfg)
		model.load_state_dict(state_dict)
		model.to(device)

		print('开始预测...')
		preds = predict(df, model, device, drug_dict, prot_dict, batch_size=args.batch_size)
		df['Pred'] = preds

		out_path = os.path.join(merge_out_dir, 'prediction_result.csv')
		final_df = df.drop(columns=['Protein'], errors='ignore')
		final_df.to_csv(out_path, index=False)
		print(f'Saved merged prediction result: {out_path}')
	else:
		for uniprot in uniprot_list:
			print(f"\nProcessing Uniprot ID: {uniprot}")
			out_dir = os.path.join(results_root, uniprot)
			os.makedirs(out_dir, exist_ok=True)

			# 1) 提取
			print('提取 unibiomap 条目...')
			df = extract_unibiomap_for_uniprots(
				uniprot_ids=[uniprot],
				save_dir=out_dir,
				save_csv_name='unibiomap_dti.csv',
				write_fasta=True,
				show_progress=False,
				conf_thres=args.conf_thres,
			)
			df = df.drop(columns=['Target_Uniprot'], errors='ignore')
			print(f'Original count: {len(df)}')
			
			# 2) 预先计算属性并过滤
			print('计算属性并过滤无效分子...')
			df = filter_and_compute_properties(df)
			
			if df.empty:
				print(f"Warning: No valid molecules for {uniprot} after filtering. Skipping.")
				continue

			# 3) 预处理
			print('预处理分子与蛋白...')
			drug_dict, prot_dict = prepare_screening_data(df, num_processors=args.num_processors)

			# 4) 模型初始化
			# 更新 deg
			cfg['deg'] = get_deg_from_list(list(drug_dict.values()))
			
			model = UnetDTI(cfg)
			model.load_state_dict(state_dict)
			model.to(device)

			# 5) 预测
			print('开始预测...')
			preds = predict(df, model, device, drug_dict, prot_dict, batch_size=args.batch_size)
			df['Pred'] = preds
			
			# 保存单体结果
			out_path = os.path.join(out_dir, 'prediction_result.csv')
			final_df = df.drop(columns=['Protein'], errors='ignore')
			final_df.to_csv(out_path, index=False)
			print(f'Saved prediction result: {out_path}')
			
			results_map[uniprot] = df

	# 6) Common logic
	if args.merge_pred and args.find_common:
		print("\nmerge-pred 模式下已跳过 find-common（合并模式仅输出一个整体结果）。")
	elif args.find_common and len(results_map) > 1:
		print("\nFinding common molecules...")
		find_common_molecules(results_map, results_root)
	elif args.find_common:
		print("\nNeed at least 2 successful results to find common molecules.")


if __name__ == '__main__':
	main()

