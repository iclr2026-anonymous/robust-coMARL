import os
import re
import glob
import pickle
from typing import Dict, List, Tuple
from datetime import datetime

import numpy as np
import torch
import matplotlib.pyplot as plt

from configs.config import EVAL_CONFIGS, get_network_params
from utils.evaluation import parallel_evaluate_policy
from networks.networks import QNetwork
from sustaingym.envs.building import MultiAgentBuildingEnv, ParameterGenerator
from environments.env_utils import normalize_action_space


# --- Configuration ---
RESULTS_ROOT = os.path.join(os.path.dirname(__file__), 'results')
DEFAULT_SEEDS = [11]

# Mapping from user-facing algorithm names to directory key used in results
ALG_NAME_TO_KEY = {
	'VDN': 'vdn',
	'QMIX': 'qmix',
	'QTRAN': 'qtran',
	'VDN_GROUPDR': 'vdn_groupdr',
	'QMIX_GROUPDR': 'qmix_groupdr',
	'QTRAN_GROUPDR': 'qtran_groupdr',
	'VDN_G': 'vdn_g',
	'QMIX_G': 'qmix_g',
	'QTRAN_G': 'qtran_g',
}

# Mapping from algorithm keys to display names
ALG_KEY_TO_DISPLAY_NAME = {
	'vdn': 'VDN-Contamination',
	'qmix': 'QMIX-Contamination',
	'qtran': 'QTRAN-Contamination',
	'vdn_g': 'VDN-TV',
	'qmix_g': 'QMIX-TV',
	'qtran_g': 'QTRAN-TV',
	'vdn_groupdr': 'VDN-GroupDR',
	'qmix_groupdr': 'QMIX-GroupDR',
	'qtran_groupdr': 'QTRAN-GroupDR'
}

PLOT_RHO_VALUES = {
	'vdn': [0.0,0.008], 
	'qmix': [0.0,0.008], 
	'qtran': [0.0,0.02], 
	'vdn_g': [0.0,0.5], 
	'qmix_g': [0.0,0.2], 
	'qtran_g': [0.0,0.2], 
}

# Regex to parse directories like robust_{alg}_rho_{rho}_{timestamp}
RUN_DIR_REGEX = re.compile(r'^robust_(?P<alg>[a-z0-9_]+)_rho_(?P<rho>[0-9.]+)_(?P<ts>\d{8}_\d{6})$')

# Regex to parse seed-specific directories like results_seeds_{seed_value}
SEED_DIR_REGEX = re.compile(r'^results_seeds_(?P<seed>\d+)$')


def _build_temp_env_for_meta(eval_config: Dict) -> Tuple[int, int, List[str]]:
	"""Instantiate an env to obtain n_agents, n_actions, and agent list."""
	env = MultiAgentBuildingEnv(
		ParameterGenerator(
			building=eval_config['building'],
			weather=eval_config['weather'],
			location=eval_config['location'],
			ac_map=eval_config['ac_map'],
			is_continuous_action=False,
		)
	)
	env = normalize_action_space(env)
	n_agents = len(env.agents)
	n_actions = env.action_spaces[env.agents[0]].n
	agent_list = list(env.agents)
	return n_agents, n_actions, agent_list


def _discover_latest_run_dirs(results_root: str, alg_key: str) -> Dict[float, str]:
	"""
	Scan results_root for directories of a given algorithm key and return a map:
	{ rho_value: latest_run_dir_abs_path }.
	"""
	if not os.path.isdir(results_root):
		return {}
	candidates = []
	for name in os.listdir(results_root):
		match = RUN_DIR_REGEX.match(name)
		if not match:
			continue
		if match.group('alg') != alg_key:
			continue
		try:
			rho_val = float(match.group('rho'))
		except ValueError:
			continue
		candidates.append((rho_val, name))
	# Keep the last (latest timestamp) per rho; sort names by timestamp component
	by_rho: Dict[float, List[str]] = {}
	for rho, dirname in candidates:
		by_rho.setdefault(rho, []).append(dirname)
	latest: Dict[float, str] = {}
	for rho, dirnames in by_rho.items():
		# Sort by timestamp lexicographically which matches format YYYYMMDD_HHMMSS
		dirnames.sort(key=lambda d: RUN_DIR_REGEX.match(d).group('ts'))
		latest[rho] = os.path.join(results_root, dirnames[-1])
	return latest


def _discover_seed_dirs(results_root: str, alg_key: str, rho_val: float) -> Dict[int, str]:
	"""
	Scan results_root for seed-specific directories and return a map:
	{ seed_value: seed_dir_abs_path }.
	Look for directories like results_seeds_{seed}/robust_{alg}_rho_{rho}_{timestamp}
	"""
	if not os.path.isdir(results_root):
		print(f"Results root directory not found: {results_root}")
		return {}
		
	print(f"Searching for seed directories in: {results_root}")
	print(f"Looking for algorithm: {alg_key}, rho: {rho_val}")
	
	seed_dirs = {}
	for name in os.listdir(results_root):
		match = SEED_DIR_REGEX.match(name)
		if not match:
			continue
		
		seed_val = int(match.group('seed'))
		seed_dir = os.path.join(results_root, name)
		
		# Look for algorithm-specific runs in this seed directory
		if os.path.isdir(seed_dir):
			for run_name in os.listdir(seed_dir):
				run_match = RUN_DIR_REGEX.match(run_name)
				if run_match and run_match.group('alg') == alg_key:
					try:
						run_rho = float(run_match.group('rho'))
						if abs(run_rho - rho_val) < 1e-6:  # Float comparison
							full_path = os.path.join(seed_dir, run_name)
							seed_dirs[seed_val] = full_path
							print(f"  Found seed {seed_val}: {full_path}")
							break
					except ValueError:
						continue
	
	print(f"Total seed directories found: {len(seed_dirs)} for {alg_key} rho={rho_val}")
	return seed_dirs


def _load_q_nets_from_run(run_dir: str, n_agents: int, obs_dim: int, n_actions: int, device: torch.device) -> List[QNetwork]:
	"""Instantiate QNetwork per agent and load state_dict from network_parameter folder."""
	net_dir = os.path.join(run_dir, 'network_parameter')
	if not os.path.isdir(net_dir):
		raise FileNotFoundError(f"network_parameter folder not found in {run_dir}")
	# Expect files like agent_0_q_network.pt, agent_1_q_network.pt, ...
	q_nets: List[QNetwork] = []
	for agent_idx in range(n_agents):
		pt_path = os.path.join(net_dir, f'agent_{agent_idx}_q_network.pt')
		if not os.path.isfile(pt_path):
			# Fallback: try glob any .pt sorted
			pt_files = sorted(glob.glob(os.path.join(net_dir, '*.pt')))
			if agent_idx < len(pt_files):
				pt_path = pt_files[agent_idx]
			else:
				raise FileNotFoundError(f"Missing checkpoint for agent {agent_idx} in {net_dir}")
		model = QNetwork(obs_dim, n_actions, hidden_dim=get_network_params()['hidden_dim']).to(device)
		state = torch.load(pt_path, map_location=device)
		model.load_state_dict(state)
		model.eval()
		q_nets.append(model)
	return q_nets




def evaluate_algorithms_and_save(algorithms: List[str], seeds: List[int] = DEFAULT_SEEDS, device_str: str = 'cpu', save_dir: str | None = None):
	"""
	Evaluate latest checkpoints for each algorithm over rho values using given seeds.
	Save results to pickle files for later plotting.
	"""
	device = torch.device(device_str)
	network_params = get_network_params()
	obs_dim = network_params['obs_dim']
	# Use the first EVAL_CONFIG to build an env for meta (n_agents/n_actions)
	n_agents, n_actions, _ = _build_temp_env_for_meta(EVAL_CONFIGS[0])
	
	if save_dir is None:
		save_dir = RESULTS_ROOT
	os.makedirs(save_dir, exist_ok=True)

	# Get config names for x-axis
	config_names = [cfg['name'] for cfg in EVAL_CONFIGS]
	
	# Store all results
	all_results = {
		'timestamp': datetime.now().isoformat(),
		'seeds': seeds,
		'config_names': config_names,
		'algorithms': {},
		'baselines': {}
	}

	# Process each algorithm separately
	for alg in algorithms:
		print(f"\n=== Evaluating Algorithm: {alg} ===")
		alg_key = ALG_NAME_TO_KEY[alg]
		# skip baseline algorithms
		if alg in ['vdn_groupdr', 'qmix_groupdr', 'qtran_groupdr']:
			continue
		rho_to_run = _discover_latest_run_dirs(RESULTS_ROOT, alg_key)
		if not rho_to_run:
			print(f"No runs found for {alg} under {RESULTS_ROOT}")
			continue
		
		# Sort rhos for consistent ordering
		sorted_rhos = sorted(rho_to_run.keys())
		print(f"Found rho values for {alg}: {sorted_rhos}")
		
		alg_results = {}
		
		# Evaluate each rho value
		for rho in sorted_rhos:
			run_dir = rho_to_run[rho]
			print(f"  Processing rho={rho} from {run_dir}")
			try:
				q_nets = _load_q_nets_from_run(run_dir, n_agents, obs_dim, n_actions, device)
			except Exception as e:
				print(f"  Skip {alg} rho={rho}: {e}")
				continue
			
			# Evaluate across all configs for this rho
			config_rewards = []
			config_errors = []
			config_raw_scores = []  # Store raw scores for each config
			
			for eval_cfg in EVAL_CONFIGS:
				# Follow _evaluate_policy: create a fixed set of episode seeds and evaluate multiple episodes per config
				seed_scores = []
				for s in seeds:
					# Build deterministic episode seeds per config using a RNG seeded by s
					rng = np.random.default_rng(s)
					arr = rng.integers(0, 200, size=network_params.get('num_eval_episodes', 50), dtype=int).tolist()
					
					res = parallel_evaluate_policy(
						q_nets, eval_cfg, device,
						num_episodes=len(arr),
						num_workers=min(4, len(arr)),
						history_len=network_params['history_len'],
						episode_seeds=arr
					)
					seed_scores.append(float(res['mean_reward']))
				
				# Aggregate across seeds for this config
				config_rewards.append(float(np.mean(seed_scores)))
				config_errors.append(float(np.std(seed_scores)))
				config_raw_scores.append(seed_scores)
			
			# Store results for this rho
			alg_results[rho] = {
				'rewards': config_rewards,
				'errors': config_errors,
				'raw_scores': config_raw_scores,
				'run_dir': run_dir
			}
		
		all_results['algorithms'][alg] = alg_results
	
	# Evaluate baseline algorithms
	baseline_algorithms = ['vdn_groupdr', 'qmix_groupdr', 'qtran_groupdr']
	print(f"\n=== Evaluating Baseline Algorithms ===")
	
	for baseline_alg in baseline_algorithms:
		baseline_rho_to_run = _discover_latest_run_dirs(RESULTS_ROOT, baseline_alg)
		if baseline_rho_to_run:
			# Use rho=0.0 if available, otherwise first available
			baseline_rho = 0.0 if 0.0 in baseline_rho_to_run else sorted(baseline_rho_to_run.keys())[0]
			baseline_run_dir = baseline_rho_to_run[baseline_rho]
			
			print(f"  Processing baseline {baseline_alg} (rho={baseline_rho})")
			try:
				baseline_q_nets = _load_q_nets_from_run(baseline_run_dir, n_agents, obs_dim, n_actions, device)
				
				# Evaluate baseline across all configs
				baseline_config_rewards = []
				baseline_config_errors = []
				baseline_raw_scores = []
				
				for eval_cfg in EVAL_CONFIGS:
					baseline_seed_scores = []
					for s in seeds:
						rng = np.random.default_rng(s)
						arr = rng.integers(0, 200, size=network_params.get('num_eval_episodes', 50), dtype=int).tolist()
						
						res = parallel_evaluate_policy(
							baseline_q_nets, eval_cfg, device,
							num_episodes=len(arr),
							num_workers=min(4, len(arr)),
							history_len=network_params['history_len'],
							episode_seeds=arr
						)
						baseline_seed_scores.append(float(res['mean_reward']))
					
					baseline_config_rewards.append(float(np.mean(baseline_seed_scores)))
					baseline_config_errors.append(float(np.std(baseline_seed_scores)))
					baseline_raw_scores.append(baseline_seed_scores)
				
				# Store baseline results
				all_results['baselines'][baseline_alg] = {
					'rewards': baseline_config_rewards,
					'errors': baseline_config_errors,
					'raw_scores': baseline_raw_scores,
					'rho': baseline_rho,
					'run_dir': baseline_run_dir
				}
				print(f"    Completed baseline: {baseline_alg}")
				
			except Exception as e:
				print(f"    Warning: Could not evaluate baseline {baseline_alg}: {e}")
	
	# Save all results to pickle file
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	results_filename = f'evaluation_results_{timestamp}.pkl'
	results_path = os.path.join(save_dir, results_filename)
	
	with open(results_path, 'wb') as f:
		pickle.dump(all_results, f)
	
	print(f"\n=== Results saved to {results_path} ===")
	print(f"Evaluated {len(all_results['algorithms'])} algorithms and {len(all_results['baselines'])} baselines")
	
	return results_path


def evaluate_seed_averaged_networks(algorithms: List[str], device_str: str = 'cpu', save_dir: str | None = None):
	"""
	Evaluate networks from multiple seeds separately, then average the results.
	For each algorithm and rho value:
	1. Load networks from each available seed directory
	2. Evaluate each seed's networks independently 
	3. Average the evaluation results across all seeds
	Uses the first seed from DEFAULT_SEEDS for evaluation episodes.
	"""
	device = torch.device(device_str)
	network_params = get_network_params()
	obs_dim = network_params['obs_dim']
	eval_seed = DEFAULT_SEEDS[0]  # Use first seed for evaluation
	
	# Use the first EVAL_CONFIG to build an env for meta (n_agents/n_actions)
	n_agents, n_actions, _ = _build_temp_env_for_meta(EVAL_CONFIGS[0])
	
	if save_dir is None:
		save_dir = RESULTS_ROOT
	os.makedirs(save_dir, exist_ok=True)

	# Get config names for x-axis
	config_names = [cfg['name'] for cfg in EVAL_CONFIGS]
	
	# Store all results
	all_results = {
		'timestamp': datetime.now().isoformat(),
		'eval_seed': eval_seed,
		'config_names': config_names,
		'algorithms': {},
		'baselines': {},
		'method': 'seed_averaged'
	}

	# Process each algorithm separately
	for alg in algorithms:
		print(f"\n=== Evaluating Seed-Averaged Algorithm: {alg} ===")
		alg_key = ALG_NAME_TO_KEY[alg]
		
		# Skip baseline algorithms
		if alg_key in ['vdn_groupdr', 'qmix_groupdr', 'qtran_groupdr']:
			continue
			
		# Get allowed rho values for this algorithm
		allowed_rhos = PLOT_RHO_VALUES.get(alg_key, [])
		
		alg_results = {}
		
		# Evaluate each rho value
		for rho in allowed_rhos:
			print(f"  Processing rho={rho} for {alg}")
			
			# Discover seed directories for this algorithm and rho
			# Look in the parent directory of RESULTS_ROOT for seed directories
			seed_search_root = os.path.dirname(RESULTS_ROOT)
			seed_dirs = _discover_seed_dirs(seed_search_root, alg_key, rho)
			
			if not seed_dirs:
				print(f"  Skip {alg} rho={rho}: No seed directories found")
				continue
			
			# Evaluate each seed's networks separately, then average results
			all_seed_results = {}
			
			for seed_val, run_dir in seed_dirs.items():
				print(f"    Evaluating seed {seed_val}")
				try:
					# Load networks for this seed
					seed_nets = _load_q_nets_from_run(run_dir, n_agents, obs_dim, n_actions, device)
				except Exception as e:
					print(f"    Skip seed {seed_val}: {e}")
					continue
				
				# Evaluate this seed's networks across all configs
				seed_config_rewards = []
				
				for eval_cfg in EVAL_CONFIGS:
					# Use a single evaluation seed (same across all seeds for consistency)
					rng = np.random.default_rng(eval_seed)
					episode_seeds = rng.integers(0, 200, size=network_params.get('num_eval_episodes', 50), dtype=int).tolist()
					
					res = parallel_evaluate_policy(
						seed_nets, eval_cfg, device,
						num_episodes=len(episode_seeds),
						num_workers=min(4, len(episode_seeds)),
						history_len=network_params['history_len'],
						episode_seeds=episode_seeds
					)
					
					seed_config_rewards.append(float(res['mean_reward']))
				
				all_seed_results[seed_val] = seed_config_rewards
			
			if not all_seed_results:
				print(f"  Skip {alg} rho={rho}: No valid seed evaluations")
				continue
			
			# Average results across all seeds for each config
			config_rewards = []
			config_errors = []
			config_raw_scores = []
			
			for config_idx in range(len(EVAL_CONFIGS)):
				# Collect rewards for this config from all seeds
				config_seed_rewards = []
				for seed_val, seed_rewards in all_seed_results.items():
					if config_idx < len(seed_rewards):
						config_seed_rewards.append(seed_rewards[config_idx])
				
				if config_seed_rewards:
					mean_reward = float(np.mean(config_seed_rewards))
					std_reward = float(np.std(config_seed_rewards))
					config_rewards.append(mean_reward)
					config_errors.append(std_reward)
					config_raw_scores.append(config_seed_rewards)
				else:
					config_rewards.append(0.0)
					config_errors.append(0.0)
					config_raw_scores.append([])
			
			# Store results for this rho
			alg_results[rho] = {
				'rewards': config_rewards,
				'errors': config_errors,
				'raw_scores': config_raw_scores,
				'num_seeds_evaluated': len(all_seed_results),
				'seed_results': all_seed_results,
				'seed_dirs': seed_dirs
			}
		
		all_results['algorithms'][alg] = alg_results
	
	# Evaluate baseline algorithms
	baseline_algorithms = ['vdn_groupdr', 'qmix_groupdr', 'qtran_groupdr']
	print(f"\n=== Evaluating Baseline Algorithms (Seed-Averaged) ===")
	
	for baseline_alg in baseline_algorithms:
		print(f"\n=== Evaluating Baseline Algorithm: {baseline_alg} ===")
		
		# Discover seed directories for baseline algorithm (typically use rho=0.0)
		baseline_seed_dirs = _discover_seed_dirs(seed_search_root, baseline_alg, 0.0)
		
		if not baseline_seed_dirs:
			print(f"  Skip {baseline_alg}: No seed directories found")
			continue
		
		# Evaluate each seed's networks separately for baseline
		all_baseline_seed_results = {}
		
		for seed_val, run_dir in baseline_seed_dirs.items():
			print(f"    Evaluating baseline seed {seed_val}")
			try:
				# Load networks for this seed
				seed_nets = _load_q_nets_from_run(run_dir, n_agents, obs_dim, n_actions, device)
			except Exception as e:
				print(f"    Skip seed {seed_val}: {e}")
				continue
			
			# Evaluate this seed's networks across all configs
			seed_config_rewards = []
			
			for eval_cfg in EVAL_CONFIGS:
				# Use the same evaluation seed for consistency
				rng = np.random.default_rng(eval_seed)
				episode_seeds = rng.integers(0, 200, size=network_params.get('num_eval_episodes', 50), dtype=int).tolist()
				
				res = parallel_evaluate_policy(
					seed_nets, eval_cfg, device,
					num_episodes=len(episode_seeds),
					num_workers=min(4, len(episode_seeds)),
					history_len=network_params['history_len'],
					episode_seeds=episode_seeds
				)
				
				seed_config_rewards.append(float(res['mean_reward']))
			
			all_baseline_seed_results[seed_val] = seed_config_rewards
		
		if not all_baseline_seed_results:
			print(f"  Skip {baseline_alg}: No valid seed evaluations")
			continue
		
		# Average baseline results across all seeds for each config
		baseline_config_rewards = []
		baseline_config_errors = []
		baseline_raw_scores = []
		
		for config_idx in range(len(EVAL_CONFIGS)):
			# Collect rewards for this config from all seeds
			config_seed_rewards = []
			for seed_val, seed_rewards in all_baseline_seed_results.items():
				if config_idx < len(seed_rewards):
					config_seed_rewards.append(seed_rewards[config_idx])
			
			if config_seed_rewards:
				mean_reward = float(np.mean(config_seed_rewards))
				std_reward = float(np.std(config_seed_rewards))
				baseline_config_rewards.append(mean_reward)
				baseline_config_errors.append(std_reward)
				baseline_raw_scores.append(config_seed_rewards)
			else:
				baseline_config_rewards.append(0.0)
				baseline_config_errors.append(0.0)
				baseline_raw_scores.append([])
		
		# Store baseline results
		all_results['baselines'][baseline_alg] = {
			'rewards': baseline_config_rewards,
			'errors': baseline_config_errors,
			'raw_scores': baseline_raw_scores,
			'num_seeds_evaluated': len(all_baseline_seed_results),
			'seed_results': all_baseline_seed_results,
			'rho': 0.0,
			'seed_dirs': baseline_seed_dirs
		}
		print(f"    Completed baseline: {baseline_alg} with {len(all_baseline_seed_results)} seeds")
	
	# Save all results to pickle file
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	results_filename = f'seed_averaged_results_{timestamp}.pkl'
	results_path = os.path.join(save_dir, results_filename)
	
	with open(results_path, 'wb') as f:
		pickle.dump(all_results, f)
	
	print(f"\n=== Seed-averaged results saved to {results_path} ===")
	print(f"Evaluated {len(all_results['algorithms'])} algorithms and {len(all_results['baselines'])} baselines by averaging results across multiple seeds")
	
	return results_path


def load_and_plot_results(results_path: str, save_dir: str | None = None):
	"""
	Load evaluation results from pickle file and create plots with improved visualization.
	"""
	if not os.path.exists(results_path):
		raise FileNotFoundError(f"Results file not found: {results_path}")
	
	print(f"Loading results from {results_path}")
	with open(results_path, 'rb') as f:
		all_results = pickle.load(f)
	
	if save_dir is None:
		save_dir = os.path.dirname(results_path)
	os.makedirs(save_dir, exist_ok=True)
	
	config_names = all_results['config_names']
	
	# Reorder config_names to move Cold_Chicago to the end
	reordered_indices = []
	cold_chicago_idx = None
	
	for i, name in enumerate(config_names):
		if 'Cold_Chicago' in name:
			cold_chicago_idx = i
		else:
			reordered_indices.append(i)
	
	# Add Cold_Chicago at the end if found
	if cold_chicago_idx is not None:
		reordered_indices.append(cold_chicago_idx)
	
	# Reorder config names and create new labels
	reordered_config_names = [config_names[i] for i in reordered_indices]
	config_labels = [f'env_{i+1}' for i in range(len(reordered_config_names))]
	
	# Set font to Times New Roman
	plt.rcParams['font.family'] = 'Times New Roman'
	plt.rcParams['font.size'] = 12
	
	# Group algorithms by base algorithm type (VDN, QMIX, QTRAN)
	algorithm_groups = {
		'VDN': [],
		'QMIX': [],
		'QTRAN': []
	}
	
	for alg, alg_results in all_results['algorithms'].items():
		alg_key = ALG_NAME_TO_KEY[alg]
		allowed_rhos = PLOT_RHO_VALUES.get(alg_key, [])
		filtered_rhos = [rho for rho in sorted(alg_results.keys()) if rho in allowed_rhos]
		if len(filtered_rhos) >= 2:  # Need both robust and non-robust
			# Determine base algorithm type
			if alg_key.startswith('vdn'):
				algorithm_groups['VDN'].append(alg)
			elif alg_key.startswith('qmix'):
				algorithm_groups['QMIX'].append(alg)
			elif alg_key.startswith('qtran'):
				algorithm_groups['QTRAN'].append(alg)
	
	# Function to scale y values
	def scale_y(y_values):
		return [(y + 9000) / 9000 for y in y_values]
	
	def scale_errors(errors, original_y):
		# Scale errors proportionally
		# the num of the seeds is 5
		return [err / (9000*np.sqrt(5)) for err in errors]
	

	def plot_unified_families(algorithm_groups):
		"""Plot all algorithm families in a unified grid layout with shared row labels and legend"""
		# Filter out empty algorithm groups
		valid_groups = {name: algs for name, algs in algorithm_groups.items() if algs}
		if not valid_groups:
			print("No algorithms with sufficient data for plotting")
			return
		
		# Determine the structure: we need to handle different uncertainty types
		# First, collect all unique algorithm types across families
		all_algorithm_types = set()
		family_algorithms = {}
		
		for family_name, algorithms in valid_groups.items():
			family_algorithms[family_name] = {}
			for alg in algorithms:
				alg_key = ALG_NAME_TO_KEY[alg]
				if alg_key.endswith('_g'):
					uncertainty_type = 'TV'
				else:
					uncertainty_type = 'contamination'
				
				if uncertainty_type not in family_algorithms[family_name]:
					family_algorithms[family_name][uncertainty_type] = []
				family_algorithms[family_name][uncertainty_type].append(alg)
				all_algorithm_types.add(uncertainty_type)
		
		# Create subplot grid: rows = uncertainty types, cols = families
		uncertainty_types = sorted(list(all_algorithm_types))  # ['TV', 'contamination']
		# Order families as VDN, QMIX, QTRAN (left to right)
		desired_order = ['VDN', 'QMIX', 'QTRAN']
		family_names = [name for name in desired_order if name in valid_groups]
		
		n_rows = len(uncertainty_types)
		n_cols = len(family_names)
		
		# Create figure with direct subplot size control
		# You can directly control individual subplot size by adjusting these parameters:
		subplot_width = 6  # Width for each subplot (in inches)
		subplot_height = 4  # Height for each subplot (in inches)
		
		# Calculate total figure size based on subplot dimensions
		total_width = subplot_width * n_cols
		total_height = subplot_height * n_rows
		
		# Create figure with exact size based on subplot dimensions
		fig, axes = plt.subplots(n_rows, n_cols, figsize=(total_width, total_height), sharex=True)
		if n_rows == 1 and n_cols == 1:
			axes = [[axes]]
		elif n_rows == 1:
			axes = [axes]
		elif n_cols == 1:
			axes = [[ax] for ax in axes]
		
		# Store legend elements globally
		global_legend_elements = []
		global_legend_labels_added = set()
		
		# Plot each combination
		for row_idx, uncertainty_type in enumerate(uncertainty_types):
			for col_idx, family_name in enumerate(family_names):
				ax = axes[row_idx][col_idx]
				
				# Check if this family has algorithms for this uncertainty type
				if (family_name not in family_algorithms or 
					uncertainty_type not in family_algorithms[family_name]):
					ax.set_visible(False)
					continue
				
				algorithms = family_algorithms[family_name][uncertainty_type]
				print(f"\n=== Plotting {family_name} - {uncertainty_type} ===")
				
				# Plot all algorithms for this family-uncertainty combination
				for alg in algorithms:
					alg_results = all_results['algorithms'][alg]
					alg_key = ALG_NAME_TO_KEY[alg]
					allowed_rhos = PLOT_RHO_VALUES.get(alg_key, [])
					
					# Plot only allowed rho values as separate curves
					sorted_rhos = sorted(alg_results.keys())
					filtered_rhos = [rho for rho in sorted_rhos if rho in allowed_rhos]
					
					# Store data for shading
					non_robust_rewards = None
					robust_rewards = None
					
					# Determine the base contamination algorithm for Non-Robust version
					base_contamination_alg = None
					if alg_key.startswith('vdn'):
						base_contamination_alg = 'VDN'
					elif alg_key.startswith('qmix'):
						base_contamination_alg = 'QMIX'
					elif alg_key.startswith('qtran'):
						base_contamination_alg = 'QTRAN'
					
					for rho in filtered_rhos:
						if rho == 0.0:
							# Use base contamination algorithm's rho=0.0 data for Non-Robust version
							if base_contamination_alg and base_contamination_alg in all_results['algorithms']:
								base_alg_results = all_results['algorithms'][base_contamination_alg]
								if 0.0 in base_alg_results:
									rewards = base_alg_results[0.0]['rewards']
									errors = base_alg_results[0.0]['errors']
								else:
									rewards = alg_results[rho]['rewards']
									errors = alg_results[rho]['errors']
							else:
								rewards = alg_results[rho]['rewards']
								errors = alg_results[rho]['errors']
						else:
							rewards = alg_results[rho]['rewards']
							errors = alg_results[rho]['errors']
						
						# Reorder rewards and errors to match config reordering
						reordered_rewards = [rewards[i] for i in reordered_indices]
						reordered_errors = [errors[i] for i in reordered_indices]
						
						# Scale the rewards and errors
						scaled_rewards = scale_y(reordered_rewards)
						scaled_errors = scale_errors(reordered_errors, reordered_rewards)
						
						# Create unified labels for legend
						if rho == 0.0:
							label = "Non-robust"	
							non_robust_rewards = scaled_rewards
							color = 'blue'
						else:
							label = "Robust (ours)"
							robust_rewards = scaled_rewards
							color = 'red'
						
						# Plot this rho curve with error bars
						line = ax.errorbar(range(len(reordered_config_names)), scaled_rewards, yerr=scaled_errors, 
									marker='o', linewidth=2, markersize=4, capsize=1,
									label=label, color=color)
						
						# Add to global legend elements only once
						if label not in global_legend_labels_added:
							global_legend_elements.append(line)
							global_legend_labels_added.add(label)
					
					# Add shaded area between robust and non-robust versions (with proper intersection handling)
					if non_robust_rewards is not None and robust_rewards is not None:
						x_vals = list(range(len(reordered_config_names)))
						
						# Find continuous segments where robust > non-robust
						segments = []
						current_segment_x = []
						current_segment_upper = []
						current_segment_lower = []
						
						for i in range(len(reordered_config_names)):
							if robust_rewards[i] > non_robust_rewards[i]:
								# Add this point to current segment
								current_segment_x.append(x_vals[i])
								current_segment_upper.append(robust_rewards[i])
								current_segment_lower.append(non_robust_rewards[i])
							else:
								# End current segment if it has points
								if current_segment_x:
									segments.append((current_segment_x.copy(), current_segment_lower.copy(), current_segment_upper.copy()))
									current_segment_x.clear()
									current_segment_upper.clear()
									current_segment_lower.clear()
						
						# Don't forget the last segment
						if current_segment_x:
							segments.append((current_segment_x.copy(), current_segment_lower.copy(), current_segment_upper.copy()))
						
						# Handle interpolation at boundaries for smoother transitions
						refined_segments = []
						for seg_x, seg_lower, seg_upper in segments:
							if not seg_x:
								continue
								
							refined_x = seg_x.copy()
							refined_lower = seg_lower.copy()
							refined_upper = seg_upper.copy()
							
							# Check if we need to add interpolated points at segment boundaries
							start_idx = seg_x[0]
							end_idx = seg_x[-1]
							
							# Add interpolated point at the start if not at boundary
							if start_idx > 0:
								prev_idx = start_idx - 1
								# Linear interpolation to find intersection point
								if robust_rewards[prev_idx] != non_robust_rewards[prev_idx]:
									# Find intersection between [prev_idx, start_idx]
									x1, y1_r, y1_n = prev_idx, robust_rewards[prev_idx], non_robust_rewards[prev_idx]
									x2, y2_r, y2_n = start_idx, robust_rewards[start_idx], non_robust_rewards[start_idx]
									
									# Solve for intersection: y1_r + t*(y2_r-y1_r) = y1_n + t*(y2_n-y1_n)
									denom = (y2_r - y1_r) - (y2_n - y1_n)
									if abs(denom) > 1e-10:  # Avoid division by zero
										t = (y1_n - y1_r) / denom
										if 0 <= t <= 1:  # Intersection within segment
											intersect_x = x1 + t * (x2 - x1)
											intersect_y = y1_r + t * (y2_r - y1_r)
											# Insert at beginning
											refined_x.insert(0, intersect_x)
											refined_lower.insert(0, intersect_y)
											refined_upper.insert(0, intersect_y)
							
							# Add interpolated point at the end if not at boundary
							if end_idx < len(reordered_config_names) - 1:
								next_idx = end_idx + 1
								# Linear interpolation to find intersection point
								if robust_rewards[end_idx] != non_robust_rewards[end_idx]:
									# Find intersection between [end_idx, next_idx]
									x1, y1_r, y1_n = end_idx, robust_rewards[end_idx], non_robust_rewards[end_idx]
									x2, y2_r, y2_n = next_idx, robust_rewards[next_idx], non_robust_rewards[next_idx]
									
									# Solve for intersection
									denom = (y2_r - y1_r) - (y2_n - y1_n)
									if abs(denom) > 1e-10:  # Avoid division by zero
										t = (y1_n - y1_r) / denom
										if 0 <= t <= 1:  # Intersection within segment
											intersect_x = x1 + t * (x2 - x1)
											intersect_y = y1_r + t * (y2_r - y1_r)
											# Append at end
											refined_x.append(intersect_x)
											refined_lower.append(intersect_y)
											refined_upper.append(intersect_y)
							
							refined_segments.append((refined_x, refined_lower, refined_upper))
						
						# Draw each segment separately
						robustness_gain_added = False
						for i, (seg_x, seg_lower, seg_upper) in enumerate(refined_segments):
							if len(seg_x) >= 2:  # Need at least 2 points for fill_between
								fill = ax.fill_between(seg_x, seg_lower, seg_upper, 
												alpha=0.3, color='green')
								# Add to global legend elements only once
								if not robustness_gain_added and "Robustness Gain" not in global_legend_labels_added:
									global_legend_elements.append(fill)
									global_legend_labels_added.add("Robustness Gain")
									robustness_gain_added = True
				
				# Add baseline algorithm for this family
				target_baseline = None
				if family_name == 'VDN':
					target_baseline = 'vdn_groupdr'
				elif family_name == 'QMIX':
					target_baseline = 'qmix_groupdr'
				elif family_name == 'QTRAN':
					target_baseline = 'qtran_groupdr'
				
				if target_baseline and target_baseline in all_results['baselines']:
					baseline_data = all_results['baselines'][target_baseline]
					rewards = baseline_data['rewards']
					errors = baseline_data['errors']
					
					# Reorder and scale baseline data
					reordered_baseline_rewards = [rewards[i] for i in reordered_indices]
					reordered_baseline_errors = [errors[i] for i in reordered_indices]
					scaled_baseline_rewards = scale_y(reordered_baseline_rewards)
					scaled_baseline_errors = scale_errors(reordered_baseline_errors, reordered_baseline_rewards)
					
					# Plot baseline
					baseline_line = ax.errorbar(range(len(reordered_config_names)), scaled_baseline_rewards, yerr=scaled_baseline_errors,
								color='gray', linestyle='--', linewidth=2, marker='s', markersize=4, 
								alpha=0.8, capsize=1, label='GroupDR (baseline)')
					
					# Add to global legend elements only once
					if "GroupDR (baseline)" not in global_legend_labels_added:
						global_legend_elements.append(baseline_line)
						global_legend_labels_added.add("GroupDR (baseline)")
				
				# Customize subplot
				ax.set_ylim(0, 1)
				ax.grid(True, alpha=0.3)
				
				# Move all y-axis to the right side
				ax.yaxis.tick_right()
				
				# Only show y-axis tick labels on the rightmost column
				if col_idx == n_cols - 1:
					# Add y-axis label to the rightmost column
					ax.yaxis.set_label_position('right')
					ax.set_ylabel('Normalized Team Reward', fontsize=14, rotation=90)
				else:
					# Hide y-axis tick labels for non-rightmost columns
					ax.tick_params(axis='y', labelleft=False, labelright=False)
				
				# Add column title only for top row
				if row_idx == 0:
					ax.set_title(f'{family_name}', fontsize=24, fontweight='bold', pad=20)
		
		# Add shared row labels
		for row_idx, uncertainty_type in enumerate(uncertainty_types):
			if uncertainty_type == 'TV':
				row_label = r'TV uncertainty'
			else:
				row_label = r'$\rho$-contamination'
			
			# Add row label on the left side of the rightmost subplot
			rightmost_col = n_cols - 1
			axes[row_idx][0].text(-0.05, 0.5, row_label, transform=axes[row_idx][0].transAxes, 
								 fontsize=24, rotation=90, verticalalignment='center', 
								 horizontalalignment='center', fontweight='bold')
		
		# Set x-axis labels only for the bottom row
		for col_idx in range(n_cols):
			if axes[-1][col_idx].get_visible():
				axes[-1][col_idx].set_xlabel(r'shift level $\rightarrow$', fontsize=24, fontweight='bold')
				axes[-1][col_idx].set_xticks(range(len(reordered_config_names)))
				axes[-1][col_idx].set_xticklabels(config_labels, rotation=0, ha='center', fontsize=14)
		
		# Create unified legend at the bottom center
		ordered_labels = ["Non-robust", "GroupDR (baseline)", "Robust (ours)", "Robustness Gain"]
		ordered_elements = []
		for label in ordered_labels:
			for element in global_legend_elements:
				element_label = element.get_label() if hasattr(element, 'get_label') else label
				if element_label == label or (label == "Robustness Gain" and not hasattr(element, 'get_label')):
					ordered_elements.append(element)
					break
		
		fig.legend(ordered_elements, ordered_labels, loc='lower center', 
				  bbox_to_anchor=(0.5, 0.1), ncol=4, fontsize=24, frameon=True, fancybox=True, shadow=True)
		
		# Simple layout adjustment - let matplotlib handle most spacing automatically
		plt.tight_layout()
		# Only adjust bottom margin for legend
		plt.subplots_adjust(bottom=0.1)
		
		# Save unified plot as PDF
		save_path = os.path.join(save_dir, 'unified_families_reward_vs_config_eval.pdf')
		plt.savefig(save_path, format='pdf', bbox_inches='tight')
		print(f"Saved unified families plot to {save_path}")
		
		# Show plot
		plt.show()
		
		# Close figure to free memory
		plt.close()
	
	# Plot unified families instead of individual families
	plot_unified_families(algorithm_groups)


def evaluate_algorithms(algorithms: List[str], seeds: List[int] = DEFAULT_SEEDS, device_str: str = 'cpu', save_dir: str | None = None, save_results: bool = True):
	"""
	Evaluate latest checkpoints for each algorithm over rho values using given seeds.
	For each algorithm: create separate plot showing different rho values and baselines.
	X-axis: eval_configs, Y-axis: episodic reward.
	
	Args:
		algorithms: List of algorithm names to evaluate
		seeds: Random seeds for evaluation
		device_str: Device to run evaluation on
		save_dir: Directory to save results and plots
		save_results: Whether to save results to pickle file
	"""
	if save_results:
		# Save results first, then plot
		results_path = evaluate_algorithms_and_save(algorithms, seeds, device_str, save_dir)
		load_and_plot_results(results_path, save_dir)
	else:
		# Legacy behavior: evaluate and plot directly (not recommended for large evaluations)
		print("Warning: Running evaluation without saving results. Consider using save_results=True for large evaluations.")
		# Here we could keep the original direct evaluation code if needed


def find_latest_results_file(save_dir: str | None = None) -> str | None:
	"""Find the most recent evaluation results file."""
	if save_dir is None:
		save_dir = RESULTS_ROOT
	
	pattern = os.path.join(save_dir, 'evaluation_results_*.pkl')
	files = glob.glob(pattern)
	
	if not files:
		return None
	
	# Sort by timestamp in filename
	files.sort(key=lambda x: os.path.basename(x).split('_')[-1].split('.')[0])
	return files[-1]


def list_results_files(save_dir: str | None = None) -> List[str]:
	"""List all available evaluation results files."""
	if save_dir is None:
		save_dir = RESULTS_ROOT
	
	pattern = os.path.join(save_dir, 'evaluation_results_*.pkl')
	files = glob.glob(pattern)
	files.sort(key=lambda x: os.path.basename(x).split('_')[-1].split('.')[0])
	return files


def plot_seed_averaged_results(results_path: str, save_dir: str | None = None):
	"""
	Load and plot evaluation results averaged across multiple seeds.
	Creates separate plots for each base algorithm (VDN, QMIX, QTRAN) showing 
	all their variants and related baselines.
	"""
	if not os.path.exists(results_path):
		raise FileNotFoundError(f"Results file not found: {results_path}")
	
	print(f"Loading seed-averaged results from {results_path}")
	with open(results_path, 'rb') as f:
		all_results = pickle.load(f)
	
	if save_dir is None:
		save_dir = os.path.dirname(results_path)
	os.makedirs(save_dir, exist_ok=True)
	
	config_names = all_results['config_names']
	
	# Reorder config_names to move Cold_Chicago to the end
	reordered_indices = []
	cold_chicago_idx = None
	
	for i, name in enumerate(config_names):
		if 'Cold_Chicago' in name:
			cold_chicago_idx = i
		else:
			reordered_indices.append(i)
	
	# Add Cold_Chicago at the end if found
	if cold_chicago_idx is not None:
		reordered_indices.append(cold_chicago_idx)
	
	# Reorder config names and create new labels
	reordered_config_names = [config_names[i] for i in reordered_indices]
	config_labels = [f'env_{i+1}' for i in range(len(reordered_config_names))]
	
	# Set font to Times New Roman
	plt.rcParams['font.family'] = 'Times New Roman'
	plt.rcParams['font.size'] = 12
	
	# Create separate plots for each individual algorithm
	for alg, alg_results in all_results['algorithms'].items():
		alg_key = ALG_NAME_TO_KEY[alg]
		display_name = ALG_KEY_TO_DISPLAY_NAME.get(alg_key, alg_key.upper())
		
		print(f"\n=== Creating plot for {display_name} ===")
		
		# Create figure for this algorithm
		plt.figure(figsize=(14, 10))
		
		colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'cyan']
		markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']
		linestyles = ['-', '--', '-.', ':']
		
		plot_idx = 0
		
		# Plot all rho values for this algorithm
		sorted_rhos = sorted(alg_results.keys())
		
		# Determine the base contamination algorithm for Non-Robust version
		base_contamination_alg = None
		if alg_key.startswith('vdn'):
			base_contamination_alg = 'VDN'
		elif alg_key.startswith('qmix'):
			base_contamination_alg = 'QMIX'
		elif alg_key.startswith('qtran'):
			base_contamination_alg = 'QTRAN'
		
		for rho in sorted_rhos:
			if rho == 0.0:
				# Use base contamination algorithm's rho=0.0 data for Non-Robust version
				if base_contamination_alg and base_contamination_alg in all_results['algorithms']:
					base_alg_results = all_results['algorithms'][base_contamination_alg]
					if 0.0 in base_alg_results:
						rewards = base_alg_results[0.0]['rewards']
						errors = base_alg_results[0.0]['errors']
						num_seeds = base_alg_results[0.0]['num_seeds_evaluated']
					else:
						# Fallback to current algorithm's data
						rewards = alg_results[rho]['rewards']
						errors = alg_results[rho]['errors']
						num_seeds = alg_results[rho]['num_seeds_evaluated']
				else:
					# Fallback to current algorithm's data
					rewards = alg_results[rho]['rewards']
					errors = alg_results[rho]['errors']
					num_seeds = alg_results[rho]['num_seeds_evaluated']
			else:
				# Use current algorithm's data for Robust version
				rewards = alg_results[rho]['rewards']
				errors = alg_results[rho]['errors']
				num_seeds = alg_results[rho]['num_seeds_evaluated']
			
			# Reorder rewards and errors to match config reordering
			reordered_rewards = [rewards[i] for i in reordered_indices]
			reordered_errors = [errors[i] for i in reordered_indices]
			
			# Create label
			rho_label = "Non-Robust" if rho == 0.0 else f"ρ={rho}"
			label = f"{display_name} {rho_label} (avg {num_seeds} seeds)"
			
			# Choose style
			color = colors[plot_idx % len(colors)]
			marker = markers[plot_idx % len(markers)]
			linestyle = linestyles[plot_idx % len(linestyles)]
			
			# Plot this algorithm-rho combination with error bars
			plt.errorbar(range(len(reordered_config_names)), reordered_rewards, yerr=reordered_errors,
						color=color, marker=marker, linestyle=linestyle,
						linewidth=2, markersize=8, capsize=3, label=label)
			
			plot_idx += 1
		
		# Add corresponding baseline algorithm for this algorithm family
		if 'baselines' in all_results:
			baseline_colors = ['gray', 'lightcoral', 'lightblue', 'lightgreen']
			baseline_styles = ['--', ':', '-.', '-']
			baseline_idx = 0
			
			# Determine which baseline to show based on algorithm family
			target_baseline = None
			if alg_key.startswith('vdn'):
				target_baseline = 'vdn_groupdr'
			elif alg_key.startswith('qmix'):
				target_baseline = 'qmix_groupdr'
			elif alg_key.startswith('qtran'):
				target_baseline = 'qtran_groupdr'
			
			# Only plot the corresponding baseline algorithm
			if target_baseline and target_baseline in all_results['baselines']:
				baseline_data = all_results['baselines'][target_baseline]
				rewards = baseline_data['rewards']
				errors = baseline_data['errors']
				num_seeds = baseline_data['num_seeds_evaluated']
				
				# Reorder baseline rewards and errors to match config reordering
				reordered_baseline_rewards = [rewards[i] for i in reordered_indices]
				reordered_baseline_errors = [errors[i] for i in reordered_indices]
				
				# Get display name for baseline algorithm
				baseline_display_name = ALG_KEY_TO_DISPLAY_NAME.get(target_baseline, target_baseline.upper())
				
				# Plot baseline with distinctive style
				color = baseline_colors[baseline_idx % len(baseline_colors)]
				style = baseline_styles[baseline_idx % len(baseline_styles)]
				label = f'{baseline_display_name} (baseline, avg {num_seeds} seeds)'
				
				plt.errorbar(range(len(reordered_config_names)), reordered_baseline_rewards, yerr=reordered_baseline_errors,
							color=color, linestyle=style, linewidth=3, marker='s', markersize=8, 
							alpha=0.8, capsize=3, label=label)
		
		# Customize plot
		plt.xlabel(r'shift level $\rightarrow$', fontsize=12)
		plt.ylabel('Normalized Team Reward', fontsize=12)
		plt.title(f'Seed-Averaged {display_name} Performance vs Configuration', fontsize=14, fontweight='bold')
		
		# Set x-axis ticks
		plt.xticks(range(len(reordered_config_names)), config_labels, rotation=0, ha='center')
		
		# Add grid and legend
		plt.grid(True, alpha=0.3)
		plt.legend(loc='lower left')
		
		# Adjust layout to prevent clipping
		plt.tight_layout()
		
		# Save plot for this algorithm
		save_path = os.path.join(save_dir, f'seed_averaged_{alg_key}_eval.png')
		plt.savefig(save_path, dpi=300, bbox_inches='tight')
		print(f"Saved {display_name} plot to {save_path}")
		
		# Show plot
		plt.show()
		
		# Close figure to free memory
		plt.close()


def export_results_to_table(results_path: str, output_file: str = 'results.txt', show_raw_data: bool = False):
	"""
	Export evaluation results to a formatted table in a text file.
	"""
	if not os.path.exists(results_path):
		raise FileNotFoundError(f"Results file not found: {results_path}")
	
	print(f"Loading results from {results_path}")
	with open(results_path, 'rb') as f:
		all_results = pickle.load(f)
	
	config_names = all_results['config_names']
	
	# Reorder config_names to move Cold_Chicago to the end (same as plotting)
	reordered_indices = []
	cold_chicago_idx = None
	
	for i, name in enumerate(config_names):
		if 'Cold_Chicago' in name:
			cold_chicago_idx = i
		else:
			reordered_indices.append(i)
	
	# Add Cold_Chicago at the end if found
	if cold_chicago_idx is not None:
		reordered_indices.append(cold_chicago_idx)
	
	# Reorder config names and create new labels
	reordered_config_names = [config_names[i] for i in reordered_indices]
	config_labels = [f'env_{i+1}' for i in range(len(reordered_config_names))]
	
	# Define scaling functions (same as in plotting)
	def scale_y(y_values):
		return [(y + 9000) / 9000 for y in y_values]
	
	def scale_errors(errors, original_y):
		return [err / (9000*np.sqrt(5)) for err in errors]
	
	# Prepare output content
	output_lines = []
	output_lines.append("="*80)
	output_lines.append("EVALUATION RESULTS TABLE")
	output_lines.append("="*80)
	output_lines.append(f"Generated on: {all_results['timestamp']}")
	
	if 'method' in all_results and all_results['method'] == 'seed_averaged':
		output_lines.append(f"Method: Seed-averaged evaluation")
		output_lines.append(f"Evaluation seed: {all_results.get('eval_seed', 'N/A')}")
	else:
		output_lines.append(f"Method: Standard evaluation")
		output_lines.append(f"Seeds used: {all_results.get('seeds', 'N/A')}")
	
	if show_raw_data:
		output_lines.append("Data format: Raw data (original rewards) and Scaled data (normalized to [0,1])")
	else:
		output_lines.append("Data format: Scaled data (normalized using (y + 9000) / 9000)")
	
	output_lines.append("")
	
	# Configuration mapping table
	output_lines.append("CONFIGURATION MAPPING:")
	output_lines.append("-" * 50)
	for i, (label, orig_name) in enumerate(zip(config_labels, reordered_config_names)):
		output_lines.append(f"{label:8} : {orig_name}")
	output_lines.append("")
	
	# Main algorithms results
	if 'algorithms' in all_results and all_results['algorithms']:
		output_lines.append("ALGORITHM PERFORMANCE RESULTS:")
		output_lines.append("=" * 50)
		
		for alg, alg_results in all_results['algorithms'].items():
			alg_key = ALG_NAME_TO_KEY[alg]
			display_name = ALG_KEY_TO_DISPLAY_NAME.get(alg_key, alg_key.upper())
			
			output_lines.append(f"\n{display_name}:")
			output_lines.append("-" * len(display_name))
			
			# Table header
			header = f"{'RHO':<8} | " + " | ".join([f"{label:>12}" for label in config_labels])
			output_lines.append(header)
			output_lines.append("-" * len(header))
			
			# Sort rho values
			sorted_rhos = sorted(alg_results.keys())
			
			for rho in sorted_rhos:
				rewards = alg_results[rho]['rewards']
				errors = alg_results[rho]['errors']
				
				# Reorder rewards and errors to match config reordering
				reordered_rewards = [rewards[i] for i in reordered_indices]
				reordered_errors = [errors[i] for i in reordered_indices]
				
				# Calculate scaled data (same as plotting)
				scaled_rewards = scale_y(reordered_rewards)
				scaled_errors = scale_errors(reordered_errors, reordered_rewards)
				
				if show_raw_data:
					# Show both raw and scaled data
					
					# Raw data row
					rho_label = f"{rho:.3f}"
					values_str = " | ".join([f"{reward:8.1f}±{error:4.1f}" for reward, error in zip(reordered_rewards, reordered_errors)])
					row = f"{rho_label:<8} | {values_str}"
					output_lines.append(row + " (Raw)")
					
					# Scaled data row
					values_str_scaled = " | ".join([f"{reward:8.3f}±{error:6.3f}" for reward, error in zip(scaled_rewards, scaled_errors)])
					row_scaled = f"{'(scaled)':<8} | {values_str_scaled}"
					output_lines.append(row_scaled)
					output_lines.append("")  # Empty line for separation
				else:
					# Show only scaled data (default)
					rho_label = f"{rho:.3f}"
					values_str_scaled = " | ".join([f"{reward:8.3f}±{error:6.3f}" for reward, error in zip(scaled_rewards, scaled_errors)])
					row = f"{rho_label:<8} | {values_str_scaled}"
					output_lines.append(row)
			
			# Add seed information if available
			if sorted_rhos and 'num_seeds_evaluated' in alg_results[sorted_rhos[0]]:
				num_seeds = alg_results[sorted_rhos[0]]['num_seeds_evaluated']
				output_lines.append(f"(Based on {num_seeds} seeds)" if 'method' in all_results and all_results['method'] == 'seed_averaged' else f"(Based on {len(all_results.get('seeds', []))} seeds)")
	
	# Baseline algorithms results
	if 'baselines' in all_results and all_results['baselines']:
		output_lines.append("\n\nBASELINE ALGORITHM RESULTS:")
		output_lines.append("=" * 50)
		
		for baseline_alg, baseline_data in all_results['baselines'].items():
			baseline_display_name = ALG_KEY_TO_DISPLAY_NAME.get(baseline_alg, baseline_alg.upper())
			
			output_lines.append(f"\n{baseline_display_name}:")
			output_lines.append("-" * len(baseline_display_name))
			
			rewards = baseline_data['rewards']
			errors = baseline_data['errors']
			
			# Reorder baseline rewards and errors to match config reordering
			reordered_baseline_rewards = [rewards[i] for i in reordered_indices]
			reordered_baseline_errors = [errors[i] for i in reordered_indices]
			
			# Calculate scaled baseline data (same as plotting)
			scaled_baseline_rewards = scale_y(reordered_baseline_rewards)
			scaled_baseline_errors = scale_errors(reordered_baseline_errors, reordered_baseline_rewards)
			
			# Table header
			header = f"{'TYPE':<8} | " + " | ".join([f"{label:>12}" for label in config_labels])
			output_lines.append(header)
			output_lines.append("-" * len(header))
			
			if show_raw_data:
				# Show both raw and scaled data for baselines
				
				# Raw baseline row
				values_str = " | ".join([f"{reward:8.1f}±{error:4.1f}" for reward, error in zip(reordered_baseline_rewards, reordered_baseline_errors)])
				row = f"{'baseline':<8} | {values_str}"
				output_lines.append(row + " (Raw)")
				
				# Scaled baseline row
				values_str_scaled = " | ".join([f"{reward:8.3f}±{error:6.3f}" for reward, error in zip(scaled_baseline_rewards, scaled_baseline_errors)])
				row_scaled = f"{'(scaled)':<8} | {values_str_scaled}"
				output_lines.append(row_scaled)
			else:
				# Show only scaled data for baselines (default)
				values_str_scaled = " | ".join([f"{reward:8.3f}±{error:6.3f}" for reward, error in zip(scaled_baseline_rewards, scaled_baseline_errors)])
				row = f"{'baseline':<8} | {values_str_scaled}"
				output_lines.append(row)
			
			# Add seed information if available
			if 'num_seeds_evaluated' in baseline_data:
				num_seeds = baseline_data['num_seeds_evaluated']
				output_lines.append(f"(Based on {num_seeds} seeds)" if 'method' in all_results and all_results['method'] == 'seed_averaged' else f"(Based on {len(all_results.get('seeds', []))} seeds)")
	
	output_lines.append("\n" + "="*80)
	output_lines.append("END OF RESULTS")
	output_lines.append("="*80)
	
	# Write to file
	output_path = os.path.join(os.path.dirname(results_path), output_file)
	with open(output_path, 'w', encoding='utf-8') as f:
		f.write('\n'.join(output_lines))
	
	print(f"Results table exported to: {output_path}")
	return output_path


def export_latest_results_to_table(save_dir: str | None = None, output_file: str = 'results.txt', show_raw_data: bool = False):
	"""Export the most recent results to a table."""
	latest_file = find_latest_results_file(save_dir)
	if latest_file is None:
		print("No evaluation results files found. Please run evaluation first.")
		return None
	
	print(f"Using latest results file: {latest_file}")
	return export_results_to_table(latest_file, output_file, show_raw_data)


def plot_from_latest_results(save_dir: str | None = None):
	"""Load and plot from the most recent results file."""
	latest_file = find_latest_results_file(save_dir)
	if latest_file is None:
		print("No evaluation results files found. Please run evaluation first.")
		return
	
	print(f"Using latest results file: {latest_file}")
	load_and_plot_results(latest_file, save_dir)


if __name__ == '__main__':
	import argparse
	
	parser = argparse.ArgumentParser(description='Evaluate saved MARL networks')
	parser.add_argument('--mode', choices=['evaluate', 'plot', 'both', 'seed_averaged', 'export_table'], default='both',
						help='Mode: evaluate (run evaluation), plot (plot from saved results), both, seed_averaged (evaluate each seed separately then average results), or export_table (export results to table)')
	parser.add_argument('--results_file', type=str, default=None,
						help='Specific results file to plot from (if mode=plot) or export from (if mode=export_table)')
	parser.add_argument('--algorithms', nargs='+', default=None,
						help='Algorithms to evaluate (default: all)')
	parser.add_argument('--device', type=str, default='cpu', help='Device to run on')
	parser.add_argument('--save_dir', type=str, default=None, help='Directory to save results')
	parser.add_argument('--output_file', type=str, default='results.txt', help='Output filename for table export')
	parser.add_argument('--show_raw_data', action='store_true', help='Show both raw and scaled data in table export (default: only scaled data)')
	
	args = parser.parse_args()
	
	if args.mode == 'evaluate':
		# Only run evaluation, save results
		algs = args.algorithms if args.algorithms else list(ALG_NAME_TO_KEY.keys())
		print(f"Running evaluation for algorithms: {algs}")
		evaluate_algorithms_and_save(algs, DEFAULT_SEEDS, args.device, args.save_dir)
		
	elif args.mode == 'plot':
		# Only plot from existing results
		if args.results_file:
			if not os.path.exists(args.results_file):
				print(f"Results file not found: {args.results_file}")
			else:
				load_and_plot_results(args.results_file, args.save_dir)
		else:
			plot_from_latest_results(args.save_dir)
	
	elif args.mode == 'seed_averaged':
		# Run seed-averaged evaluation and plotting
		algs = args.algorithms if args.algorithms else list(ALG_NAME_TO_KEY.keys())
		print(f"Running seed-averaged evaluation for algorithms: {algs}")
		results_path = evaluate_seed_averaged_networks(algs, args.device, args.save_dir)
		plot_seed_averaged_results(results_path, args.save_dir)
		
	elif args.mode == 'export_table':
		# Export results to table
		if args.results_file:
			if not os.path.exists(args.results_file):
				print(f"Results file not found: {args.results_file}")
			else:
				export_results_to_table(args.results_file, args.output_file, args.show_raw_data)
		else:
			export_latest_results_to_table(args.save_dir, args.output_file, args.show_raw_data)
			
	else:  # mode == 'both'
		# Default behavior: evaluate and plot
		algs = args.algorithms if args.algorithms else list(ALG_NAME_TO_KEY.keys())
		print(f"Running evaluation and plotting for algorithms: {algs}")
		evaluate_algorithms(algs, DEFAULT_SEEDS, args.device, args.save_dir, save_results=True)
