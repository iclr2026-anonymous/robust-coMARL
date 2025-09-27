# Robust Cooperative Multi-Agent Reinforcement Learning (RobustCoopMARL)

An implementation of robust cooperative multi-agent reinforcement learning algorithms including VDN, QMIX, and QTRAN with various robustness techniques for multi-agent building energy management.

## Overview

This project implements several robust MARL algorithms designed to handle environmental uncertainties in multi-agent systems. The algorithms are tested on building energy management tasks using the SustainGym environment.

### Supported Algorithms

- **VDN**: Basic value decomposition approach
- **QMIX**: Monotonic value function factorization
- **QTRAN**: General value function factorization
- **G-Network variants** (`_g` suffix): TV (Total Variation) uncertainty-based robust versions
- **Contamination variants**: ρ-contamination robust versions  
- **GroupDR variants**: Group domain randomization baselines
- **Domain Randomization** (`vdn_dr`): Standard domain randomization to learn the behavior policy for GroupDR algorithms


## Quick Setup

### Prerequisites

- Python 3.11 or higher

### 1. Create Virtual Environment

It's recommended to use a virtual environment to manage dependencies:

**Using venv:**
```bash
python3.11 -m venv RobustMARL
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**Using conda:**
```bash
conda create --name robust_marl python=3.11
conda activate robust_marl
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Training Algorithms

Use the main training script to train different algorithms:

```bash
python main.py --algorithm <algorithm_name> [--rho <rho_value>] [--device <device>]
```

#### Available Algorithms:
- `vdn` - Standard VDN
- `qmix` - Standard QMIX  
- `qtran` - Standard QTRAN
- `vdn_g` - VDN with G-network (TV uncertainty)
- `qmix_g` - QMIX with G-network
- `qtran_g` - QTRAN with G-network
- `vdn_groupdr` - VDN with GroupDR
- `qmix_groupdr` - QMIX with GroupDR
- `qtran_groupdr` - QTRAN with GroupDR
- `vdn_dr` - VDN with domain randomization
- `env_estimator` - Train environment estimator

#### Examples:

**Train VDN with default rho values:**
```bash
python main.py --algorithm vdn
```

**Train QMIX with specific rho value:**
```bash
python main.py --algorithm qmix --rho 0.008
```

**Train VDN G-network variant:**
```bash
python main.py --algorithm vdn_g --rho 0.2
```

**Train environment estimator (required for GroupDR methods):**
```bash
python main.py --algorithm env_estimator --behavior_checkpoint_dir results_seeds_0/behavior/
```

**Train GroupDR method:**
```bash
python main.py --algorithm vdn_groupdr --env_estimator_path results_seeds_0/robust_env_estimator_rho_0.0_*/env_estimator.pt
```

#### Command Line Arguments:
- `--algorithm`: Algorithm to train (required)
- `--rho`: Specific rho value for robustness (optional, defaults to algorithm-specific values)
- `--device`: Device to use (`cuda` or `cpu`, optional, auto-detects if not specified)
- `--behavior_checkpoint_dir`: For env_estimator, path to VDN run directory with network parameters
- `--env_estimator_path`: For GroupDR methods, path to trained environment estimator .pt file

### Evaluating Trained Models

Use the evaluation script to assess trained models:

```bash
python evaluate_saved_networks.py --mode <mode> [--algorithms <alg1> <alg2> ...] [--device <device>]
```

#### Evaluation Modes:
- `evaluate` - Run evaluation and save results
- `plot` - Plot from saved results  
- `both` - Run evaluation and plot (default)
- `seed_averaged` - Evaluate across multiple seeds and average
- `export_table` - Export results to text table

#### Examples:

**Evaluate all algorithms:**
```bash
python evaluate_saved_networks.py --mode both
```

**Evaluate specific algorithms:**
```bash
python evaluate_saved_networks.py --mode both --algorithms VDN QMIX
```

**Generate seed-averaged results:**
```bash
python evaluate_saved_networks.py --mode seed_averaged --algorithms VDN QMIX QTRAN
```

**Export results to table:**
```bash
python evaluate_saved_networks.py --mode export_table --output_file my_results.txt
```

## Configuration

The `configs/config.py` file contains all configuration parameters:

- **Environment configs**: Training and evaluation environment settings
- **Training parameters**: Episodes, learning rates, network architectures
- **Algorithm parameters**: Algorithm-specific hyperparameters
- **Robustness parameters**: Rho values for different uncertainty types

Key configuration sections:
- `TRAINING_CONFIG`: Base training environment
- `EVAL_CONFIGS`: List of evaluation environments for generalization testing
- `RHO_VALUES`: Robustness parameter ranges for each algorithm (for training)
- `TRAINING_PARAMS`: Training hyperparameters per algorithm
- `NETWORK_PARAMS`: Neural network architecture parameters

## Results

Training results are automatically saved to timestamped directories in `results/` or `results_seeds_<seed>/`:

```
results_seeds_0/
├── robust_vdn_rho_0.0_20250918_123456/
│   ├── network_parameter/          # Saved model weights
│   │   ├── agent_0_q_network.pt
│   │   ├── agent_1_q_network.pt
│   │   └── ...
│   ├── results.pkl                 # Training metrics
│   └── training_curve.png          # Training curve plot
└── ...
```

Evaluation results are saved as pickle files with comprehensive performance metrics across different environment configurations.
