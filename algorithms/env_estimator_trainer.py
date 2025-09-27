import os
import pickle
from typing import List, Optional

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from configs.config import EVAL_CONFIGS, get_training_params, get_network_params, get_algorithm_params, create_results_dir, create_log_file
from environments.env_utils import create_environment, process_obs_dict_partial
from networks import QNetwork, EnvEstimatorWithStep
from utils.evaluation import select_actions
from utils.replay_buffer import ReplayBuffer

class EnvEstimatorTrainer:
	"""
	Train EnvEstimator to predict per-environment immediate reward given (state, joint_action).
	Behavior policy is fixed by loading saved VDN Q-network parameters.
	"""
	def __init__(self, behavior_checkpoint_dir: str, device: Optional[torch.device] = None):
		self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.training_params = get_training_params('env_estimator')
		self.network_params = get_network_params()
		self.algorithm_params = get_algorithm_params('env_estimator')

		# set random seed
		self._set_random_seed()

		# rng for random seed
		self.action_rng = np.random.default_rng(self.training_params['random_seed'])
		self.buffer_rng = np.random.default_rng(self.training_params['batch_seed'])
		self.env_rng = np.random.default_rng(self.training_params['env_seed'])

		# Load fixed behavior Q-networks from saved directory
		self.q_nets: List[QNetwork] = self._load_behavior_policy(behavior_checkpoint_dir)
		self.n_agents = len(self.q_nets)
		self.n_actions = self._infer_n_actions()

		# Initialize EnvEstimator
		self.env_estimator = EnvEstimatorWithStep(
			state_dim=self.network_params['state_dim'],
			n_agents=self.n_agents,
			n_actions=self._infer_n_actions(),
			hidden_dim=self.network_params['hidden_dim']
		).to(self.device)
		self.optimizer = torch.optim.Adam(self.env_estimator.parameters(), lr=self.algorithm_params.get('learning_rate', 1e-3))
		self.criterion = nn.MSELoss()

		# Replay buffer and training state
		self.buffer = ReplayBuffer(self.network_params['buffer_capacity'],self.buffer_rng)
		self.global_step = 0
		self.train_losses: list[float] = []

		# Fixed epsilon for exploration
		self.epsilon = 0.5

	def _infer_n_actions(self) -> int:
		# Infer from one of the behavior nets
		return self.q_nets[0].n_actions

	def _compute_epsilon(self):
		"""Return fixed epsilon value for exploration."""
		return self.epsilon

	def _set_random_seed(self):
		"""set random seed to ensure reproducible results"""
		import random
		import os
        
        # get seed from config
		seed = getattr(self.training_params, 'random_seed', 42)
        
        # set random seed
		random.seed(seed)
		np.random.seed(seed)
		torch.manual_seed(seed)
        
        # if CUDA is available, set CUDA random seed
		if torch.cuda.is_available():
			torch.cuda.manual_seed(seed)
			torch.cuda.manual_seed_all(seed)
			torch.backends.cudnn.deterministic = True
			torch.backends.cudnn.benchmark = False
        
        # set environment variable
		os.environ['PYTHONHASHSEED'] = str(seed)
        
		print(f"Random seed set to {seed}")

	def _select_actions_epsilon_greedy(self, obs_histories, action_histories, agent_list):
		"""Select actions using epsilon-greedy policy."""
		epsilon = self._compute_epsilon()
		
		if self.action_rng.random() < epsilon:
			# Random exploration
			return [self.action_rng.integers(self.n_actions) for _ in agent_list]
		else:
			# Greedy exploitation using fixed behavior policy
			return select_actions(self.q_nets, obs_histories, action_histories, agent_list, self.device, self.network_params['history_len'])

	def _load_behavior_policy(self, checkpoint_dir: str) -> List[QNetwork]:
		# Expect files agent_{i}_q_network.pt inside network_parameter
		network_param_dir = os.path.join(checkpoint_dir, 'network_parameter')
		if not os.path.isdir(network_param_dir):
			raise FileNotFoundError(f"network_parameter not found in {checkpoint_dir}")
		# Find agent param files
		agent_files = sorted([f for f in os.listdir(network_param_dir) if f.startswith('agent_') and f.endswith('_q_network.pt')])
		if not agent_files:
			raise FileNotFoundError(f"No agent_*_q_network.pt found in {network_param_dir}")
		# Create dummy env to know n_agents and obs_dim, but we assume obs_dim fixed from config
		obs_dim = self.network_params['obs_dim']
		# For action dim we cannot know without env, but model file does not encode it; we will infer after load by reading weight shape
		q_nets: List[QNetwork] = []
		for file in agent_files:
			state_dict = torch.load(os.path.join(network_param_dir, file), map_location=self.device)
			# Infer n_actions from last layer shape
			for k, v in state_dict.items():
				if k.endswith('fc_out.weight'):
					n_actions = v.shape[0]
					break
			else:
				raise ValueError("Could not infer n_actions from state_dict")
			q_net = QNetwork(obs_dim, n_actions, hidden_dim=self.network_params['hidden_dim']).to(self.device)
			q_net.load_state_dict(state_dict)
			q_net.eval()
			q_nets.append(q_net)
		return q_nets

	def _maybe_update(self):
		if not self.buffer.is_ready(self.network_params['batch_size']):
			return None
		if self.global_step % self.training_params['update_freq'] != 0:
			return None
		batch = self.buffer.sample(self.network_params['batch_size'])
		states = torch.FloatTensor(np.stack([b[0] for b in batch], axis=0)).to(self.device)
		joint_actions = torch.LongTensor(np.stack([b[1] for b in batch], axis=0)).to(self.device)
		env_indices = torch.LongTensor(np.array([b[2] for b in batch], dtype=np.int64)).to(self.device)
		rewards = torch.FloatTensor(np.array([b[3] for b in batch], dtype=np.float32)).to(self.device)
		timesteps = torch.LongTensor(np.array([b[4] for b in batch], dtype=np.int64)).to(self.device)/255.0
		preds = self.env_estimator(states, joint_actions, timesteps)
		preds_selected = preds.gather(1, env_indices.view(-1, 1)).squeeze(1)
		
		loss = self.criterion(preds_selected, rewards)
		
		self.optimizer.zero_grad()
		loss.backward()
		torch.nn.utils.clip_grad_norm_(self.env_estimator.parameters(), self.network_params['grad_clip_norm'])
		self.optimizer.step()
		self.train_losses.append(loss.item())
		if self.global_step % 1000 == 0:
			self._log_training_progress(loss.item(), preds_selected, rewards)
		return loss.item()

	def _episode_rollout(self, eval_idx: int, episode):
		# Create env from selected config
		eval_config = EVAL_CONFIGS[eval_idx]
		env = create_environment(eval_config)
		agent_list = list(env.agents)
		rng = np.random.default_rng(self.training_params['env_seed'])
		seeds = rng.integers(0, 200, size=self.training_params['num_episodes'], dtype=int)
		seeds = seeds.tolist()
		episode_seed = seeds[episode]
		obs_dict, _ = env.reset(episode_seed)
		obs_dict_partial = process_obs_dict_partial(obs_dict, env)

		obs_histories = {agent: [obs_dict_partial[agent]] for agent in agent_list}
		action_histories = {agent: [0] for agent in agent_list}
		state = obs_dict[0]
		done = False

		t = 0
		while not done:
			# Epsilon-greedy actions from fixed behavior
			actions = self._select_actions_epsilon_greedy(obs_histories, action_histories, agent_list)
			joint_action = np.array(actions, dtype=np.int64)
			action_dict = {agent: act for agent, act in zip(agent_list, actions)}
			next_obs_dict, rewards, dones, _, _ = env.step(action_dict)
			reward = float(sum(rewards.values()))
			done = any(dones.values())
			if done:
				break
			# next
			next_obs_dict_partial = process_obs_dict_partial(next_obs_dict, env)
			for agent in agent_list:
				obs_histories[agent].append(next_obs_dict_partial[agent])
				action_histories[agent].append(int(actions[agent_list.index(agent)]))
				if len(obs_histories[agent]) > self.network_params['history_len']:
					obs_histories[agent].pop(0)
					action_histories[agent].pop(0)
			# add to replay buffer: (state, joint_action, env_idx, reward, timestep)
			self.buffer.add((state.copy(), joint_action.copy(), eval_idx, reward, int(t)))
			self.global_step += 1
			# periodic update
			self._maybe_update()
			# advance state
			state = next_obs_dict[0]
			t += 1

	def _log_training_progress(self, loss: float,preds,target):
		"""Log training progress to file."""
		log_file = create_log_file('env_estimator')
		with open(log_file, 'a') as f:
			f.write(f"[Step {self.global_step}] loss: {loss:.6f}\n")
			f.write(f"[Step {self.global_step}] preds: mean={preds.mean().item():.4f}, min={preds.min().item():.4f}, max={preds.max().item():.4f}\n")
			f.write(f"[Step {self.global_step}] target: mean={target.mean().item():.4f}, min={target.min().item():.4f}, max={target.max().item():.4f}\n")

	def train(self):
		pbar = tqdm(range(self.training_params['num_episodes']), desc='Training EnvEstimator')
		for episode in pbar:
			# random sample an environment index each episode
			eval_idx = self.env_rng.integers(0, len(EVAL_CONFIGS))
			self._episode_rollout(eval_idx,episode)

		return self._save_results()

	def _save_results(self):
		# Create results dir and save model
		results_dir = create_results_dir('env_estimator', rho_value=0.0)
		os.makedirs(results_dir, exist_ok=True)
		model_path = os.path.join(results_dir, 'env_estimator.pt')
		torch.save(self.env_estimator.state_dict(), model_path)
		# Save training curve data
		with open(os.path.join(results_dir, 'results.pkl'), 'wb') as f:
			pickle.dump({'train_losses': self.train_losses}, f)
		return {'results_dir': results_dir, 'model_path': model_path} 
