import os
import json
import shutil
from typing import Tuple, List, Dict

import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import QED, Descriptors, Lipinski
from tqdm import tqdm
from joblib import Parallel, delayed

# 导入项目中的构建器（与原有脚本一致）
from dataset.databuild import from_smiles
from dataset.databuild_dti import easy_data, integer_label_protein


def extract_unibiomap_for_uniprot(uniprot_id: str, save_dir: str, conf_thres: float = 0.9) -> pd.DataFrame:
	"""兼容旧接口：提取单个 uniprot。"""
	result_df = extract_unibiomap_for_uniprots(
		[uniprot_id],
		save_dir=save_dir,
		save_csv_name='unibiomap_dti.csv',
		write_fasta=True,
		show_progress=False,
		conf_thres=conf_thres,
	)

	if 'Target_Uniprot' in result_df.columns:
		result_df = result_df[result_df['Target_Uniprot'] == uniprot_id].copy()
		result_df = result_df.drop(columns=['Target_Uniprot'], errors='ignore')

	return result_df


def extract_unibiomap_for_uniprots(
	uniprot_ids: List[str],
	save_dir: str = None,
	save_csv_name: str = 'unibiomap_dti.csv',
	write_fasta: bool = False,
	show_progress: bool = True,
	conf_thres: float = 0.9,
) -> pd.DataFrame:
	"""从 unibiomap.pred.full.csv 中提取指定 uniprot 的 DTI 预测，保存为 unibiomap_dti.csv

	支持一次性提取一批 uniprot，复杂度近似 O(N + K)：
	- N: unibiomap 记录数（单次扫描）
	- K: 目标蛋白数（用于可选 fasta 写出）

	返回列: UCI, SMILES, Protein, Conf, Target_Uniprot
	"""
	if not uniprot_ids:
		return pd.DataFrame(columns=['UCI', 'SMILES', 'Protein', 'Conf', 'Target_Uniprot'])

	target_ids = [str(x).strip() for x in uniprot_ids if str(x).strip()]
	target_ids = list(dict.fromkeys(target_ids))
	if not target_ids:
		return pd.DataFrame(columns=['UCI', 'SMILES', 'Protein', 'Conf', 'Target_Uniprot'])

	src = os.path.join('dataset', 'data', 'unibiomap', 'unibiomap.pred.full.csv')
	if not os.path.exists(src):
		raise FileNotFoundError(f"未找到文件: {src}")

	df = pd.read_csv(src)
	if show_progress:
		tqdm.write(f"Loaded unibiomap rows: {len(df)}")

	mask = (
		((df['htype'] == 'protein') & (df['ttype'] == 'compound') & (df['h'].isin(target_ids))) |
		((df['htype'] == 'compound') & (df['ttype'] == 'protein') & (df['t'].isin(target_ids)))
	)
	filtered = df[mask].copy()
	filtered = filtered[filtered['conf'] >= conf_thres].copy()
	if filtered.empty:
		if save_dir:
			os.makedirs(save_dir, exist_ok=True)
			pd.DataFrame(columns=['UCI', 'SMILES', 'Protein', 'Conf', 'Target_Uniprot']).to_csv(
				os.path.join(save_dir, save_csv_name), index=False
			)
		return pd.DataFrame(columns=['UCI', 'SMILES', 'Protein', 'Conf', 'Target_Uniprot'])

	comp_json = json.load(open(os.path.join('dataset', 'data', 'unibiomap', 'compound_desc.json'), 'r'))
	prot_json = json.load(open(os.path.join('dataset', 'data', 'unibiomap', 'protein_desc.json'), 'r'))

	uci2smi = {k: v.get('smiles', '') for k, v in comp_json.items()}
	uniprot2seq = {k: v.get('sequence', '') for k, v in prot_json.items()}

	pairs = filtered.copy()
	pairs['protein_id'] = pairs['h'].where(pairs['htype'] == 'protein', pairs['t'])
	pairs['compound_id'] = pairs['t'].where(pairs['htype'] == 'protein', pairs['h'])
	pairs = pairs[['protein_id', 'compound_id', 'conf']]
	pairs = pairs.groupby(['protein_id', 'compound_id'], as_index=False)['conf'].max()

	pairs['SMILES'] = pairs['compound_id'].map(uci2smi)
	pairs['Protein'] = pairs['protein_id'].map(uniprot2seq).fillna('')
	pairs['Conf'] = pairs['conf']
	pairs['Target_Uniprot'] = pairs['protein_id']

	result_df = pairs[['compound_id', 'SMILES', 'Protein', 'Conf', 'Target_Uniprot']].rename(
		columns={'compound_id': 'UCI'}
	)

	if save_dir:
		os.makedirs(save_dir, exist_ok=True)
		out_path = os.path.join(save_dir, save_csv_name)
		result_df.to_csv(out_path, index=False)

		if write_fasta:
			iter_ids = tqdm(target_ids, desc='Writing FASTA') if show_progress else target_ids
			for uid in iter_ids:
				seq = uniprot2seq.get(uid, '')
				seq_path = os.path.join(save_dir, f'{uid}.fasta')
				with open(seq_path, 'w') as fasta:
					fasta.write(f">{uid}\n")
					fasta.write(f"{seq}\n")

	return result_df


def prepare_screening_data(df: pd.DataFrame, num_processors: int = 4) -> Tuple[dict, dict]:
	"""将 SMILES 和蛋白序列预处理为模型输入所需的数据结构。

	返回 (drug_dict, prot_dict)
	drug_dict: smiles -> graph/data (与 databuild 生成的对象一致)
	prot_dict: protein_seq -> torch.tensor 序列表示
	"""
	smiles_list = list(dict.fromkeys(df['SMILES'].tolist()))
	protein_list = list(dict.fromkeys(df['Protein'].tolist()))

	mols = [Chem.MolFromSmiles(s) for s in tqdm(smiles_list, desc='smiles2mol')]

	ed = easy_data(from_smiles, get_fp=False, with_hydrogen=False, with_coordinate=False, seed=123)
	data_list = list(Parallel(n_jobs=num_processors)(
		delayed(ed.process)(smi, mol) for smi, mol in tqdm(zip(smiles_list, mols), total=len(smiles_list), desc='Generating drug graphs')
	))

	drug_dict = {data.smiles: data for data in data_list}

	prot_dict = {}
	for seq in tqdm(protein_list, desc='seq2embd'):
		prot_dict[seq] = torch.tensor(integer_label_protein(seq, 1200), dtype=torch.float32)

	return drug_dict, prot_dict


def filter_and_compute_properties(df: pd.DataFrame) -> pd.DataFrame:
	"""计算 QED 与 Lipinski 违规指标，并过滤无效分子（无法转mol或单原子），返回包含新列的 DataFrame。
	这里会直接删除无效的行。

	新列包括: QED, Lipinski_violations, MW, ALOGP, HBA, HBD, PSA, ROTB, AROM, ALERTS
	"""
	print(f"Initial molecule count: {len(df)}")

	feature_cols = ['QED', 'Lipinski_violations', 'MW', 'ALOGP', 'HBA', 'HBD', 'PSA', 'ROTB', 'AROM', 'ALERTS']
	if df.empty:
		empty_df = df.iloc[0:0].copy()
		for col in feature_cols:
			empty_df[col] = None
		return empty_df

	# 仅对唯一 SMILES 计算一次属性，然后映射回原表，避免重复分子重复计算
	valid_smiles_series = df['SMILES'].dropna().astype(str).str.strip()
	unique_smiles = list(dict.fromkeys([s for s in valid_smiles_series.tolist() if s]))
	print(f"Unique SMILES count: {len(unique_smiles)}")

	props_by_smiles = {}
	for smi in tqdm(unique_smiles, total=len(unique_smiles), desc='Computing properties on unique SMILES'):
		mol = Chem.MolFromSmiles(smi)
		if mol is None or mol.GetNumAtoms() < 2:
			continue

		try:
			props = QED.properties(mol)
			qed_score = float(QED.qed(mol))

			mw = Descriptors.MolWt(mol)
			hbd = Lipinski.NumHDonors(mol)
			hba = Lipinski.NumHAcceptors(mol)
			logp = Descriptors.MolLogP(mol)
			rotb = Lipinski.NumRotatableBonds(mol)

			violations = int(mw > 500) + int(hbd > 5) + int(hba > 10) + int(logp > 5) + int(rotb > 10)

			props_by_smiles[smi] = {
				'QED': qed_score,
				'Lipinski_violations': int(violations),
				'MW': float(props.MW),
				'ALOGP': float(props.ALOGP),
				'HBA': float(props.HBA),
				'HBD': float(props.HBD),
				'PSA': float(props.PSA),
				'ROTB': float(props.ROTB),
				'AROM': float(props.AROM),
				'ALERTS': float(props.ALERTS),
			}
		except Exception:
			continue

	if not props_by_smiles:
		print("No valid molecules found after filtering.")
		# 返回带列名的空 DataFrame
		empty_df = df.iloc[0:0].copy()
		for col in feature_cols:
			empty_df[col] = None
		return empty_df

	valid_mask = df['SMILES'].astype(str).str.strip().isin(props_by_smiles)
	filtered_df = df.loc[valid_mask].copy().reset_index(drop=True)

	feat_df = filtered_df['SMILES'].astype(str).str.strip().map(props_by_smiles).apply(pd.Series)
	feat_df = feat_df[feature_cols].reset_index(drop=True)

	out = pd.concat([filtered_df, feat_df], axis=1)
	print(f"Filtered molecule count: {len(out)}")
	
	return out


def find_common_molecules(results_dict: Dict[str, pd.DataFrame], results_root: str):
	"""
	results_dict: {uniprot_id: dataframe} 
	results_root: root directory to save results
	"""
	if not results_dict:
		return

	ids = list(results_dict.keys())
	if len(ids) < 2:
		return

	# Use the first one as base
	base_id = ids[0]
	common_df = results_dict[base_id].copy()
	
	# Rename columns for the base
	# We assume common_df has 'Conf' and 'Pred'.
	common_df = common_df.rename(columns={'Conf': f'Conf_{base_id}', 'Pred': f'Pred_{base_id}'})
	
	# We iterate over others and merge on SMILES
	for other_id in ids[1:]:
		other_df = results_dict[other_id]
		# Prepare other_df for merge
		# We only need SMILES and the score columns.
		other_subset = other_df[['SMILES', 'Conf', 'Pred']].rename(
			columns={'Conf': f'Conf_{other_id}', 'Pred': f'Pred_{other_id}'}
		)
		
		# Merge inner
		common_df = pd.merge(common_df, other_subset, on='SMILES', how='inner')
	
	if common_df.empty:
		print("\n" + "#"*60)
		print(f"   NO COMMON MOLECULES FOUND FOR {ids}")
		print("#"*60 + "\n")
		return

	# If we have common molecules
	dir_name = "-".join(ids)
	save_path = os.path.join(results_root, dir_name)
	os.makedirs(save_path, exist_ok=True)
	
	print(f"\nFound {len(common_df)} common molecules for {ids}")
	print(f"Saving common results to {save_path}")
	
	# Save fasta files
	for uid in ids:
		src_fasta = os.path.join(results_root, uid, f'{uid}.fasta')
		dst_fasta = os.path.join(save_path, f'{uid}.fasta')
		if os.path.exists(src_fasta):
			shutil.copy(src_fasta, dst_fasta)
			
	# Save csv
	out_csv = os.path.join(save_path, 'prediction_result.csv')
	
	# Reorder columns: UCI, SMILES, Conf_..., Pred_..., then properties
	cols = ['UCI', 'SMILES']
	conf_cols = [f'Conf_{uid}' for uid in ids]
	pred_cols = [f'Pred_{uid}' for uid in ids]
	
	# Properties columns
	prop_cols = ['QED', 'Lipinski_violations', 'MW', 'ALOGP', 
				 'HBA', 'HBD', 'PSA', 'ROTB', 'AROM', 'ALERTS']
	
	# Ensure columns exist
	final_cols = []
	for c in cols + conf_cols + pred_cols + prop_cols:
		if c in common_df.columns:
			final_cols.append(c)
	
	final_df = common_df[final_cols]
	final_df.to_csv(out_csv, index=False)
	print(f"Saved: {out_csv}")


