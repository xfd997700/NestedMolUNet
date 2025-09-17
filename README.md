# MolUNet++: Adaptive-grained Explicit Substructure and Interaction Aware Molecular Representation Learning

**MolUNet++** is a novel graph neural network that introduces nested U-Net architecture for molecular representation learning. Our approach combines hierarchical graph pooling with skip connections to capture multi-scale molecular features, enabling superior performance across diverse drug discovery tasks including molecular property prediction, drug-target interaction, and drug-drug interaction prediction.

The key innovation lies in the nested U-Net design that preserves both local atomic details and global molecular structure through hierarchical encoding-decoding processes. This architecture, coupled with a two-stage pre-training framework and molecular fingerprint integration, achieves state-of-the-art results on standard benchmarks while providing interpretable molecular substructure analysis.

## Table of Contents

- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Usage](#usage)
- [Script Parameters](#script-parameters)
- [Datasets](#datasets)
- [Configuration](#configuration)
- [Citation](#citation)
- [License](#license)


## Installation

### Prerequisites
- Python 3.8+
- CUDA 11.0+ (for GPU support)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/xfd997700/NestedMolUNet.git
cd NestedMolUNet
```

2. Install dependencies:
```bash
# Create conda environment (recommended)
conda create -n molunet python=3.8
conda activate molunet

# Install all required packages
pip install -r requirements.txt
```

**Note**: The requirements.txt includes PyTorch with CUDA 11.8 support. For different CUDA versions, please modify the torch installation accordingly.

## Data Preparation

### Dataset Structure
Place your raw data files in the following structure:
```
dataset/data/
├── pretrain/
│   ├── zinc2m.csv          # ZINC dataset for pre-training
│   └── chembl.csv          # ChEMBL dataset for pre-training
├── property/
│   ├── BBBP.csv           # Blood-brain barrier penetration
│   ├── bace.csv           # BACE dataset
│   ├── clintox.csv        # Clinical toxicity
│   └── ...                # Other property datasets
├── DTI/
│   ├── bindingdb/         # BindingDB dataset files
│   ├── biosnap/           # BioSNAP dataset files  
│   └── human/             # Human protein targets
└── DDI/
    ├── drugbank.csv       # DrugBank interactions
    └── drugbank.tab       # DrugBank processed data
```

### Process Datasets
Run the appropriate data building scripts:

```bash
cd dataset

# Process property prediction datasets
python databuild_property.py

# Process DTI datasets  
python databuild_dti.py --dataset bindingdb --num_processor 8

# Process DDI datasets
python databuild_ddi.py --dataset drugbank --neg_ent 1 --num_processor 8

# Build pre-training data (optional)
python databuild_pretrain.py --weighted_mask_rate 0.2 --motif_mask_rate 0.15
```


## Usage

### Pre-training
Two-stage pre-training for enhanced molecular representation learning:

```bash
# Stage 1: Self-supervised learning
python pretrain_stage1.py --seed 114514 --lr 1e-4 --epochs 50 --patience 8

# Stage 2: Enhanced pre-training with fingerprints
python pretrain_stage2.py --seed 114514 --lr 1e-4 --epochs 100 --patience 15
```

### Property Prediction
```bash
python benchmark_property.py \
    --dataset BBBP \
    --runs 5 \
    --epochs 500 \
    --pretrained \
    --lr_reduce_patience 5
```
Supported datasets: `BBBP`, `bace`, `clintox`, `tox21`, `sider`, `Lipophilicity`, `freesolv`, `esol`, `ld50`

### Drug-Target Interaction
```bash
python benchmark_dti.py \
    --dataset bindingdb \
    --split random \
    --runs 5 \
    --epochs 100
```
Available splits: `random`, `cluster`

### Drug-Drug Interaction
```bash
python benchmark_ddi.py \
    --dataset drugbank \
    --split S1 \
    --runs 5 \
    --epochs 30
```
Available splits: `S1`, `S2`

### Visualization
Generate molecular visualizations with attribution analysis:

```bash
# Property prediction visualization
python visual_demo_property.py --smiles "CCO" --name "ethanol" --task lipo

# DTI visualization
python visual_demo_dti.py --sequence "PROTEIN_SEQUENCE" --smiles "CCO" --name "compound"

# DDI visualization  
python visual_demo_ddi.py --h_smiles "CCO" --h_name "drug1" --t_smiles "CC(C)O" --t_name "drug2"
```

## Script Parameters

### Pre-training Scripts

#### `pretrain_stage1.py`
```bash
python pretrain_stage1.py [OPTIONS]
```
- `--seed` (float, default=114514): Random seed
- `--lr` (float, default=1e-4): Learning rate
- `--lr_reduce_rate` (float, default=0.5): Learning rate reduction rate
- `--lr_reduce_patience` (int, default=3): Learning rate reduction patience
- `--decay` (float, default=0): Weight decay
- `--patience` (int, default=8): Early stopping patience
- `--epochs` (int, default=50): Maximum training epochs
- `--min_epochs` (int, default=1): Minimum training epochs
- `--log_train_results` (flag): Whether to evaluate training set in each epoch

#### `pretrain_stage2.py`
```bash
python pretrain_stage2.py [OPTIONS]
```
- `--seed` (float, default=114514): Random seed
- `--lr` (float, default=1e-4): Learning rate
- `--lr_reduce_rate` (float, default=0.5): Learning rate reduction rate
- `--lr_reduce_patience` (int, default=4): Learning rate reduction patience
- `--decay` (float, default=1e-4): Weight decay
- `--patience` (int, default=15): Early stopping patience
- `--epochs` (int, default=100): Maximum training epochs
- `--min_epochs` (int, default=1): Minimum training epochs
- `--log_train_results` (flag): Whether to evaluate training set in each epoch

### Benchmark Scripts

#### `benchmark_property.py`
```bash
python benchmark_property.py [OPTIONS]
```
- `--dataset` (str, required): Dataset name, choices: ['BBBP', 'bace', 'clintox', 'tox21', 'sider', 'Lipophilicity', 'freesolv', 'esol', 'ld50']
- `--seed` (float, default=114514): Random seed
- `--runs` (int, default=5): Number of independent runs
- `--lr_reduce_rate` (float, default=0.5): Learning rate reduction rate
- `--lr_reduce_patience` (int, default=5): Learning rate reduction patience
- `--decay` (float, default=0): Weight decay
- `--epochs` (int, default=500): Maximum training epochs
- `--log_train_results` (flag): Whether to evaluate training set each epoch
- `--pretrained` (flag): Whether to load pretrained model

#### `benchmark_dti.py`
```bash
python benchmark_dti.py [OPTIONS]
```
- `--dataset` (str, default='bindingdb'): Dataset name
- `--split` (str, default='random'): Split mode, choices: ['random', 'cluster']
- `--runs` (int, default=1): Number of independent runs
- `--lr_reduce_rate` (float, default=0.5): Learning rate reduction rate
- `--lr_reduce_patience` (int, default=50): Learning rate reduction patience
- `--decay` (float, default=0): Weight decay
- `--epochs` (int, default=100): Maximum training epochs
- `--log_train_results` (flag): Whether to evaluate training set each epoch

#### `benchmark_ddi.py`
```bash
python benchmark_ddi.py [OPTIONS]
```
- `--dataset` (str, default='drugbank'): Dataset name
- `--split` (str, default='S1'): Split mode, choices: ['S1', 'S2']
- `--runs` (int, default=1): Number of independent runs
- `--lr_reduce_rate` (float, default=0.5): Learning rate reduction rate
- `--lr_reduce_patience` (int, default=3): Learning rate reduction patience
- `--decay` (float, default=5e-4): Weight decay
- `--epochs` (int, default=30): Maximum training epochs
- `--log_train_results` (flag): Whether to evaluate training set each epoch

### Visualization Scripts

#### `visual_demo_property.py`
```bash
python visual_demo_property.py [OPTIONS]
```
- `--smiles` (str, required): Molecular SMILES string
- `--name` (str, required): Molecular name for output files
- `--task` (str, required): Property prediction task, choices: ['lipo', 'ld50']

#### `visual_demo_dti.py`
```bash
python visual_demo_dti.py [OPTIONS]
```
- `--sequence` (str, required): Protein sequence
- `--smiles` (str, required): Molecular SMILES string
- `--name` (str, required): Name for output files

#### `visual_demo_ddi.py`
```bash
python visual_demo_ddi.py [OPTIONS]
```
- `--h_smiles` (str, required): Head molecule SMILES string
- `--h_name` (str, required): Head molecule name
- `--t_smiles` (str, required): Tail molecule SMILES string
- `--t_name` (str, required): Tail molecule name

### Data Building Scripts

#### `dataset/databuild_property.py`
```bash
python dataset/databuild_property.py
```
Processes property prediction datasets automatically (no parameters required).

#### `dataset/databuild_pretrain.py`
```bash
python dataset/databuild_pretrain.py [OPTIONS]
```
- `--weighted_mask_rate` (float, default=0.2): Weighted masking rate for atoms
- `--weighted_mask_edge` (flag, default=True): Whether to mask edges in weighted masking
- `--motif_mask_rate` (float, default=0.15): Motif masking rate
- `--motif_mask_edge` (flag, default=True): Whether to mask edges in motif masking
- `--fpSize` (int, default=1024): Fingerprint size
- `--num_processor` (float, default=24): Number of processors for parallel processing
- `--seed` (float, default=123): Random seed

#### `dataset/databuild_dti.py`
```bash
python dataset/databuild_dti.py [OPTIONS]
```
- `--dataset` (-d, str, default='human'): Dataset to preprocess, choices: ['bindingdb', 'human', 'biosnap']
- `--num_processor` (-n_p, int, default=8): Number of processors for multiprocessing

#### `dataset/databuild_ddi.py`
```bash
python dataset/databuild_ddi.py [OPTIONS]
```
- `--dataset` (-d, str, default='drugbank'): Dataset to preprocess, choices: ['drugbank', 'twosides']
- `--neg_ent` (-n, int, default=1): Number of negative samples
- `--seed` (-s, int, default=114514): Random seed
- `--test_ratio` (-t_r, float, default=0.2): Test set ratio
- `--n_folds` (-n_f, int, default=3): Number of folds
- `--num_processor` (-n_p, int, default=8): Number of processors for multiprocessing

## Datasets

The model supports various molecular datasets:

### Property Prediction
- **BBBP**: Blood-brain barrier penetration
- **BACE**: β-secretase inhibition
- **ClinTox**: Clinical toxicity
- **Tox21**: Toxicity in 21 targets
- **SIDER**: Side effects
- **Lipophilicity**: Octanol-water partition coefficient
- **FreeSolv**: Hydration free energy
- **ESOL**: Aqueous solubility
- **LD50**: Acute toxicity

### Drug-Target Interaction
- **BindingDB**: Binding affinity database
- **BioSNAP**: Biological network datasets
- **Human**: Human protein targets

### Drug-Drug Interaction
- **DrugBank**: Drug interaction database

## Configuration

Model hyperparameters can be adjusted in `config.yaml`:

```yaml
model:
  hidden_dim: 256
  num_pool_layer: 3
  jump: all

pool:
  threshold: 0.5
  act: sigmoid

MP:
  num_mp_layer: 1
  method: PNA
  heads: 2

FP:
  query_fp: true
  in_dim: 1489
  hidden_dims: [256, 256, 256]
```

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{xu2025molunet++,
  title={MolUNet++: Adaptive-grained Explicit Substructure and Interaction Aware Molecular Representation Learning},
  author={Xu, Fanding and Yang, Zhiwe and Su, Wu and Wang, Lizhuo and Meng, Deyu and Long, Jiangang},
  journal={bioRxiv},
  pages={2025--07},
  year={2025},
  publisher={Cold Spring Harbor Laboratory}
}
```

## License

This project is licensed under the [`CC BY-NC 4.0`](http://creativecommons.org/licenses/by-nc/4.0/) International License. 

You are free to:
- **Share** — copy and redistribute the material in any medium or format
- **Adapt** — remix, transform, and build upon the material

Under the following terms:
- **Attribution** — You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.
- **NonCommercial** — You may not use the material for commercial purposes.

