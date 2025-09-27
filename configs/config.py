"""
Unified configuration file for Robust MARL algorithms.
Contains all shared configuration parameters and constants.
"""

import os
from datetime import datetime
import random

# --- Environment Configurations ---
TRAINING_CONFIG = {
    'name': 'Training_Config',
    'weather': 'Hot_Dry',
    'location': 'Tucson',
    'building': 'OfficeSmall',
    'ac_map': 1
}

# List of evaluation configurations for generalization testing
EVAL_CONFIGS = [
    TRAINING_CONFIG,
    {
        'name': 'Hot_Humid_Miami',
        'weather': 'Hot_Humid',
        'location': 'Tampa',
        'building': 'OfficeSmall',
        'ac_map': 1
    },
    {
        'name': 'Cold_Chicago',
        'weather': 'Mixed_Humid',
        'location': 'NewYork',
        'building': 'OfficeSmall',
        'ac_map': 1
    },
    {
        'name': 'Very_Hot_Humid_Honolulu',
        'weather': 'Very_Hot_Humid',
        'location': 'Honolulu',
        'building': 'OfficeSmall',
        'ac_map': 1
    },
    {
        'name': 'Warm_Dry_ElPaso',
        'weather': 'Warm_Dry',
        'location': 'ElPaso',
        'building': 'OfficeSmall',
        'ac_map': 1
    },
    {
        'name': 'Cool_Marine_Seattle',
        'weather': 'Cool_Marine',
        'location': 'Seattle',
        'building': 'OfficeSmall',
        'ac_map': 1
    }
]

DR_CONFIGS = [
    TRAINING_CONFIG,
    {
        'name': 'Hot_Humid_Miami',
        'weather': 'Hot_Humid',
        'location': 'Tampa',
        'building': 'OfficeSmall',
        'ac_map': 1
    },
    {
        'name': 'Very_Hot_Humid_Honolulu',
        'weather': 'Very_Hot_Humid',
        'location': 'Honolulu',
        'building': 'OfficeSmall',
        'ac_map': 1
    },
    {
        'name': 'Warm_Dry_ElPaso',
        'weather': 'Warm_Dry',
        'location': 'ElPaso',
        'building': 'OfficeSmall',
        'ac_map': 1
    },
    {
        'name': 'Cool_Marine_Seattle',
        'weather': 'Cool_Marine',
        'location': 'Seattle',
        'building': 'OfficeSmall',
        'ac_map': 1
    }
]
# Randomly sample one environment config from all defined configs (including TRAINING_CONFIG)
def sample_env_config_from_all(seed=None):
    rng = random.Random(seed) if seed is not None else random
    return rng.choice(EVAL_CONFIGS)

# --- Training Hyperparameters ---
# Rho values for robust training
RHO_VALUES = {
    'vdn': [0.0, 0.004, 0.008, 0.012, 0.016, 0.02],
    'qmix': [0.0,0.004, 0.008, 0.012, 0.016, 0.02],
    'qtran': [0.0,0.004, 0.008, 0.012, 0.016, 0.02],
    'vdn_g': [0.0,0.1,0.2,0.3,0.4,0.5],
    'qmix_g': [0.0,0.1,0.2,0.3,0.4,0.5],
    'qtran_g': [0.0,0.1,0.2,0.3,0.4,0.5],
    'qmix_groupdr': [0.0]
}

# Training parameters
TRAINING_PARAMS = {
    'vdn': {
        'num_episodes': 600,
        'eval_freq': 600,
        'num_eval_episodes': 1,
        'update_freq': 2,
        'target_update_freq': 25000,
        'env_seed': 42,
        'batch_seed': 42,
        'random_seed': 42
    },
    'qmix': {
        'num_episodes': 600,
        'eval_freq': 600,
        'num_eval_episodes': 1,
        'update_freq': 2,
        'target_update_freq': 25000,
        'env_seed': 42,
        'batch_seed': 0,
        'random_seed': 42
    },
    'qtran': {
        'num_episodes': 600,
        'eval_freq': 600,
        'num_eval_episodes': 1,
        'update_freq': 2,
        'target_update_freq': 25000,
        'env_seed': 42,
        'batch_seed': 0,
        'random_seed': 42
    },
    'vdn_g': {
        'num_episodes': 600,
        'eval_freq': 600,
        'num_eval_episodes': 1,
        'update_freq': 8,
        'target_update_freq': 25000,
        'env_seed': 42,
        'batch_seed': 42,
        'random_seed': 42
    },
    'qmix_g': {
        'num_episodes': 600,
        'eval_freq': 600,
        'num_eval_episodes': 1,
        'update_freq': 8,
        'target_update_freq': 25000,
        'random_seed': 42,
        'env_seed': 42,
        'batch_seed': 0
    },
    'qtran_g': {
        'num_episodes': 600,
        'eval_freq': 600,
        'num_eval_episodes': 1,
        'update_freq': 8,
        'target_update_freq': 25000,
        'env_seed': 42,
        'batch_seed': 0,
        'random_seed': 42
    },
    'env_estimator': {
        'num_episodes': 1000,
        'update_freq': 2,
        'num_eval_episodes': 0,
        'target_update_freq': 1,
        'env_seed': 42,
        'batch_seed': 0,
        'random_seed': 42
    },
    'vdn_groupdr': {
        'num_episodes': 600,
        'eval_freq': 600,
        'num_eval_episodes': 1,
        'update_freq': 2,
        'target_update_freq': 25000,
        'random_seed': 42,
        'env_seed': 42,
        'batch_seed': 0
    },
    'qmix_groupdr': {
        'num_episodes': 600,
        'eval_freq': 600,
        'num_eval_episodes': 1,
        'update_freq': 2,
        'target_update_freq': 25000,
        'env_seed': 42,
        'batch_seed': 0,
        'random_seed': 42
    },
    'qtran_groupdr': {
        'num_episodes': 600,
        'eval_freq': 600,
        'num_eval_episodes': 1,
        'update_freq': 2,
        'target_update_freq': 25000,
        'env_seed': 42,
        'batch_seed': 0,
        'random_seed': 42
    },
    'vdn_dr': {
        'num_episodes': 600,
        'eval_freq': 600,
        'num_eval_episodes': 1,
        'update_freq': 2,
        'target_update_freq': 25000,
        'env_seed': 42,
        'batch_seed': 42,
        'random_seed': 42
    }
}

# --- Network Architecture Parameters ---
NETWORK_PARAMS = {
    'obs_dim': 5,
    'state_dim': 10,
    'history_len': 8,
    'hidden_dim': 64,
    'mixing_embed_dim': 32,
    'batch_size': 64,
    'buffer_capacity': 10000,
    'gamma': 0.997,
    'epsilon_start': 1.0,
    'epsilon_end': 0.01,
    'epsilon_decay': 120000,
    'learning_rate': {
        'q_networks': 0.0005,
        'g_network': 0.001,
        'mixer': 0.0005
    },
    'optimizer': {
        'q_networks': 'RMSprop',
        'g_network': 'Adam',
        'mixer': 'RMSprop'
    },
    'grad_clip_norm': 10.0
}

# --- Algorithm-specific Parameters ---
ALGORITHM_PARAMS = {
    'vdn_g': {
        'number_update_g_steps': 4,
        'number_update_q_steps': 4,
        'randomize_env_from_configs': False,
        'randomize_per_episode': True
    },
    'qmix_g': {
        'number_update_g_steps': 4,
        'number_update_q_steps': 4
    },
    'qtran_g': {
        'number_update_g_steps': 4,
        'number_update_q_steps': 4
    },
    'env_estimator': {
        'learning_rate': 0.0005
    }
}

# --- File Paths ---
RESULTS_DIR = 'results'
LOG_DIR = 'logs'

# --- Utility Functions ---
def get_training_params(algorithm):
    """Get training parameters for a specific algorithm."""
    return TRAINING_PARAMS.get(algorithm, TRAINING_PARAMS['vdn'])

def get_rho_values(algorithm):
    """Get rho values for a specific algorithm."""
    return RHO_VALUES.get(algorithm, [0.0])

def get_network_params():
    """Get network architecture parameters."""
    return NETWORK_PARAMS

def get_algorithm_params(algorithm):
    """Get algorithm-specific parameters."""
    return ALGORITHM_PARAMS.get(algorithm, {})

def create_results_dir(algorithm, rho_value):
    """Create results directory for a specific algorithm and rho value."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    dir_name = f'robust_{algorithm}_rho_{rho_value}_{timestamp}'
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return os.path.join(RESULTS_DIR, dir_name)

def create_log_file(algorithm):
    """Create log file for a specific algorithm."""
    os.makedirs(LOG_DIR, exist_ok=True)
    return os.path.join(LOG_DIR, f'{algorithm}_output.txt') 