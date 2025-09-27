"""
QTRAN algorithm implementation with contamination robustness.
"""

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import pickle

from configs.config import TRAINING_CONFIG, EVAL_CONFIGS, DR_CONFIGS, get_training_params, get_network_params, create_results_dir, create_log_file
from environments.env_utils import create_environment, process_obs_dict_partial, pad_history, pad_action_history
from networks.networks import QNetworkWithPhi, JointActionValueNetwork, StateValueNetwork, EnvEstimatorWithStep
from utils.replay_buffer import ReplayBuffer
from utils.evaluation import select_actions, parallel_evaluate_policy

class QTRANGroupDR:
    """
    QTRAN GroupDR algorithm implementation with EnvEstimator for per-step reward.
    """
    
    def __init__(self, env_estimator_path: str, device=None):
        """
        Initialize QTRAN GroupDR algorithm.
        
        Args:
            env_estimator_path: Path to env_estimator .pt file
            device: Device to run networks on
        """
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Get configuration parameters
        self.training_params = get_training_params('qtran_groupdr')
        self.network_params = get_network_params()
        
        # set random seed
        self._set_random_seed()

		# rng for random seed
        self.action_rng = np.random.default_rng(self.training_params['random_seed'])
        self.buffer_rng = np.random.default_rng(self.training_params['batch_seed'])
        self.env_rng = np.random.default_rng(self.training_params['env_seed'])

        # Create environment
        self.env = create_environment(TRAINING_CONFIG)
        self.agent_info = self._get_agent_info()
        
        # Initialize networks
        self.q_nets, self.target_nets = self._initialize_networks()
        self.joint_q_net, self.target_joint_net = self._initialize_joint_networks()
        self.v_net = self._initialize_v_network()
        self.optimizer = self._initialize_optimizer()
        self.env_estimator = self._initialize_env_estimator(env_estimator_path)
        
        # Initialize replay buffer
        self.buffer = ReplayBuffer(self.network_params['buffer_capacity'],self.buffer_rng)
        
        # Training state
        self.global_step = 0
        self.train_rewards = []
        self.eval_checkpoints = {}

    
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

    def _get_agent_info(self):
        """Get agent information from environment."""
        return {
            'n_agents': len(self.env.agents),
            'agent_list': list(self.env.agents),
            'obs_dims': [self.network_params['obs_dim'] for _ in self.env.agents],
            'n_actions': self.env.action_spaces[self.env.agents[0]].n
        }
    
    def _initialize_env_estimator(self, path: str) -> EnvEstimatorWithStep:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"EnvEstimator weights not found at: {path}")
        estimator = EnvEstimatorWithStep(
			state_dim=self.network_params['state_dim'],
			n_agents=self.agent_info['n_agents'],
			n_actions=self.agent_info['n_actions'],
			hidden_dim=self.network_params['hidden_dim']
		).to(self.device)
        state_dict = torch.load(path, map_location=self.device)
        estimator.load_state_dict(state_dict)
        estimator.eval()
        for p in estimator.parameters():
            p.requires_grad = False
        return estimator
    
    def _initialize_networks(self):
        """Initialize Q-networks and target networks."""
        q_nets = [
            QNetworkWithPhi(obs_dim, self.agent_info['n_actions'], 
                           hidden_dim=self.network_params['hidden_dim']).to(self.device)
            for obs_dim in self.agent_info['obs_dims']
        ]
        
        target_nets = [
            QNetworkWithPhi(obs_dim, self.agent_info['n_actions'], 
                           hidden_dim=self.network_params['hidden_dim']).to(self.device)
            for obs_dim in self.agent_info['obs_dims']
        ]
        
        # Initialize target networks with Q-network weights
        for q_net, target_net in zip(q_nets, target_nets):
            target_net.load_state_dict(q_net.state_dict())
            
        return q_nets, target_nets
    
    def _initialize_joint_networks(self):
        """Initialize joint action-value networks."""
        joint_q_net = JointActionValueNetwork(
            self.agent_info['n_agents'], 
            self.agent_info['n_actions'], 
            hidden_dim=64
        ).to(self.device)
        
        target_joint_net = JointActionValueNetwork(
            self.agent_info['n_agents'], 
            self.agent_info['n_actions'], 
            hidden_dim=64
        ).to(self.device)
        
        target_joint_net.load_state_dict(joint_q_net.state_dict())
        return joint_q_net, target_joint_net
    
    def _initialize_v_network(self):
        """Initialize state value network."""
        return StateValueNetwork(
            self.network_params['state_dim'],
            self.agent_info['n_agents'], 
            hidden_dim=64
        ).to(self.device)
    
    def _initialize_optimizer(self):
        """Initialize optimizer for all networks."""
        all_params = list(self.joint_q_net.parameters())
        for q_net in self.q_nets:
            all_params += list(q_net.parameters())
        all_params += list(self.v_net.parameters())
        
        return torch.optim.Adam(all_params, lr=0.001)
    
    def _is_valid_transition(self, transition):
        """Check if a transition tuple is valid."""
        if len(transition) != 10:
            return False
        obs_histories, action_histories, actions, _, next_obs_histories, next_action_histories, _, state, next_state, _ = transition
        if len(obs_histories) != self.agent_info['n_agents'] or len(next_obs_histories) != self.agent_info['n_agents']:
            return False
        if len(action_histories) != self.agent_info['n_agents'] or len(next_action_histories) != self.agent_info['n_agents']:
            return False
        if len(actions) != self.agent_info['n_agents']:
            return False
        return True
    
    def _compute_epsilon(self):
        """Compute current epsilon value for epsilon-greedy exploration."""
        epsilon_start = self.network_params['epsilon_start']
        epsilon_end = self.network_params['epsilon_end']
        epsilon_decay = self.network_params['epsilon_decay']
        
        return max(epsilon_end, epsilon_start - (epsilon_start - epsilon_end) * self.global_step / epsilon_decay)
    
    def _select_actions(self, obs_histories, action_histories):
        """Select actions using epsilon-greedy policy."""
        epsilon = self._compute_epsilon()
        
        if self.action_rng.random() < epsilon:
            return [self.action_rng.integers(self.agent_info['n_actions']) for _ in self.agent_info['agent_list']]
        else:
            return select_actions(
                self.q_nets, obs_histories, action_histories, 
                self.agent_info['agent_list'], self.device, 
                self.network_params['history_len']
            )
    
    def _store_transition(self, obs_histories, action_histories, actions, reward, 
                         next_obs_histories, next_action_histories, done, state, next_state,timestep):
        """Store transition in replay buffer."""
        transition = (
            [pad_history(obs_histories[agent], self.network_params['history_len'], 
                        obs_dim=self.network_params['obs_dim']) for agent in self.agent_info['agent_list']],
            [pad_action_history(action_histories[agent], self.network_params['history_len']) 
             for agent in self.agent_info['agent_list']],
            actions.copy() if isinstance(actions, list) else actions,
            reward,
            [pad_history(next_obs_histories[agent], self.network_params['history_len'], 
                        obs_dim=self.network_params['obs_dim']) for agent in self.agent_info['agent_list']],
            [pad_action_history(next_action_histories[agent], self.network_params['history_len']) 
             for agent in self.agent_info['agent_list']],
            done,
            state,
            next_state,
            timestep
        )
        
        if self._is_valid_transition(transition):
            self.buffer.add(transition)
    
    def _update_networks(self):
        """Update all networks using experience replay."""
        if not self.buffer.is_ready(self.network_params['batch_size']):
            return
            
        batch = self.buffer.sample(self.network_params['batch_size'])
        
        # Prepare batch data
        obs_batch = [
            torch.FloatTensor(np.stack([transition[0][i] for transition in batch])).to(self.device)
            for i in range(self.agent_info['n_agents'])
        ]
        action_histories_batch = [
            torch.FloatTensor(np.stack([transition[1][i] for transition in batch])).to(self.device)
            for i in range(self.agent_info['n_agents'])
        ]
        actions_batch = torch.tensor([transition[2] for transition in batch], dtype=torch.long).to(self.device)
        reward_batch = torch.tensor([transition[3] for transition in batch], dtype=torch.float).to(self.device)
        next_obs_batch = [
            torch.FloatTensor(np.stack([transition[4][i] for transition in batch])).to(self.device)
            for i in range(self.agent_info['n_agents'])
        ]
        next_action_histories_batch = [
            torch.FloatTensor(np.stack([transition[5][i] for transition in batch])).to(self.device)
            for i in range(self.agent_info['n_agents'])
        ]
        done_batch = torch.tensor([transition[6] for transition in batch], dtype=torch.float).to(self.device)
        state_batch = torch.FloatTensor(np.stack([transition[7] for transition in batch])).to(self.device)
        reward_batch_envmin = self._compute_min_estimated_reward(batch)
        # Compute Q-values and hidden features for current observations
        all_q_outputs = [
            q_net(obs_batch[i], action_histories_batch[i].unsqueeze(-1))
            for i, q_net in enumerate(self.q_nets)
        ]
        all_q_values = [output[0] for output in all_q_outputs]
        cached_hidden_features = [output[1] for output in all_q_outputs]
        
        # Get V-value
        v_jt = self.v_net(state_batch, cached_hidden_features).squeeze(-1)
        
        # Gather Q-values for chosen actions
        chosen_q_values = []
        chosen_actions = []
        best_actions = []
        best_q_values = []
        
        for i, q_values in enumerate(all_q_values):
            if actions_batch.dim() > 1:
                agent_actions = actions_batch[:, i].unsqueeze(-1)
            else:
                agent_actions = torch.zeros(actions_batch.size(0), 1, dtype=torch.long, device=self.device)
                for j, acts in enumerate([b[2] for b in batch]):
                    agent_actions[j, 0] = acts[i]
            
            chosen_actions.append(agent_actions.squeeze(-1))
            chosen_q = q_values.gather(1, agent_actions).squeeze(-1)
            chosen_q_values.append(chosen_q)
            best_actions.append(q_values.max(dim=1)[1])
            best_q_values.append(q_values.max(dim=1)[0])
        
        chosen_actions_stack = torch.stack(chosen_actions, dim=1)
        chosen_q_stack = torch.stack(chosen_q_values, dim=1)
        best_actions_stack = torch.stack(best_actions, dim=1)
        best_q_stack = torch.stack(best_q_values, dim=1)
        
        # Sum Q-values across agents
        prime_q_jt_1 = torch.sum(best_q_stack, dim=1)
        prime_q_jt_2 = torch.sum(chosen_q_stack, dim=1)
        
        # Get joint Q-values
        q_jt, _ = self.joint_q_net(actions_batch, cached_hidden_features)
        q_jt = q_jt.squeeze(-1)
        hat_q_jt_1, _ = self.joint_q_net(best_actions_stack, cached_hidden_features)
        hat_q_jt_1 = hat_q_jt_1.detach().squeeze(-1)
        hat_q_jt_2, _ = self.joint_q_net(chosen_actions_stack, cached_hidden_features)
        hat_q_jt_2 = hat_q_jt_2.detach().squeeze(-1)
        
        # Compute target Q-values for next observations
        with torch.no_grad():
            next_q_outputs = [
                target_net(next_obs_batch[i], next_action_histories_batch[i].unsqueeze(-1))
                for i, target_net in enumerate(self.target_nets)
            ]
            next_q_values = [output[0] for output in next_q_outputs]
            next_actions_batch = [q_values.max(dim=1)[1] for q_values in next_q_values]
            next_joint_actions_batch = torch.stack(next_actions_batch, dim=1)
            next_cached_hidden = [output[1] for output in next_q_outputs]
            next_q_jt, _ = self.target_joint_net(next_joint_actions_batch, next_cached_hidden)
            next_q_jt = next_q_jt.squeeze(-1)
            target = reward_batch_envmin + self.network_params['gamma'] * next_q_jt * (1 - done_batch)
        
        # Compute QTRAN losses
        L_td = torch.mean(torch.square(q_jt - target))  # TD loss
        L_opt = torch.mean(torch.square(prime_q_jt_1 - hat_q_jt_1 + v_jt))  # Optimality loss
        L_nopt = torch.mean(torch.square(torch.minimum(prime_q_jt_2 - hat_q_jt_2 + v_jt, torch.zeros_like(prime_q_jt_2))))  # Non-optimality loss
        
        # Total loss
        loss = L_td + L_opt + L_nopt
        
        # Update networks
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.optimizer.param_groups[0]['params'], 
                                     max_norm=self.network_params['grad_clip_norm'])
        self.optimizer.step()
        
        # Log training progress
        if self.global_step % 1000 == 0:
            self._log_training_progress(loss, L_td, L_opt, L_nopt)

    def _compute_min_estimated_reward(self, batch):
		# Build tensors for EnvEstimator: state_batch, joint_actions_batch
        state_batch = torch.FloatTensor(np.stack([transition[7] for transition in batch])).to(self.device)
        if isinstance(batch[0][2], list):
            joint_actions_np = np.stack([np.array(transition[2], dtype=np.int64) for transition in batch], axis=0)
        else:
            joint_actions_np = np.stack([transition[2] for transition in batch], axis=0).astype(np.int64)
        joint_actions_batch = torch.LongTensor(joint_actions_np).to(self.device)
		# timesteps are stored at index 9
        timesteps = torch.LongTensor(np.array([transition[9] for transition in batch], dtype=np.int64)).to(self.device)/288
        with torch.no_grad():
            pred_rewards = self.env_estimator(state_batch, joint_actions_batch, timesteps)  # (batch, num_envs)
            min_rewards, _ = torch.min(pred_rewards, dim=1)
        return min_rewards
    
    def _log_training_progress(self, loss, L_td, L_opt, L_nopt):
        """Log training progress to file."""
        log_file = create_log_file('qtran_groupdr')
        with open(log_file, 'a') as f:
            f.write(f"[Step {self.global_step}] loss: {loss.item():.6f}\n")
            f.write(f"[Step {self.global_step}] L_td: {L_td.item():.6f}, L_opt: {L_opt.item():.6f}, L_nopt: {L_nopt.item():.6f}\n")
    
    def _update_target_networks(self):
        """Update target networks."""
        if self.global_step % self.training_params['target_update_freq'] == 0:
            for q_net, target_net in zip(self.q_nets, self.target_nets):
                target_net.load_state_dict(q_net.state_dict())
            self.target_joint_net.load_state_dict(self.joint_q_net.state_dict())
            print(f"Target networks updated at step {self.global_step}")
    
    def train(self):
        """Main training loop."""
        rng = np.random.default_rng(self.training_params['env_seed'])
        seeds = rng.integers(0, 200, size=self.training_params['num_episodes'], dtype=int)
        seeds = seeds.tolist()
        pbar = tqdm(range(self.training_params['num_episodes']), 
                   desc=f'Training QTRAN GroupDR (rho={0.0})')
        
        for episode in pbar:
            # random sample an environment index each episode
            eval_idx = self.env_rng.integers(0, len(DR_CONFIGS))
            eval_config = DR_CONFIGS[eval_idx]
            self.env = create_environment(eval_config)
            episode_seed = seeds[episode]
            obs_dict, _ = self.env.reset(episode_seed)
            obs_dict_partial = process_obs_dict_partial(obs_dict, self.env)
            
            obs_histories = {agent: [obs_dict_partial[agent]] for agent in self.agent_info['agent_list']}
            action_histories = {agent: [0] for agent in self.agent_info['agent_list']}
            
            done = False
            episode_reward = 0
            state = obs_dict[0]  # Global state
            t = 0
            
            while not done:
                actions = self._select_actions(obs_histories, action_histories)
                action_dict = {agent: act for agent, act in zip(self.agent_info['agent_list'], actions)}
                
                next_obs_dict, rewards, dones, _, _ = self.env.step(action_dict)
                next_state = next_obs_dict[0]  # Next global state
                reward = sum(rewards.values())
                done = any(dones.values())
                
                if done:
                    break
                
                # Process next observations and histories
                next_obs_dict_partial = process_obs_dict_partial(next_obs_dict, self.env)
                next_obs_histories = {agent: hist.copy() for agent, hist in obs_histories.items()}
                next_action_histories = {agent: hist.copy() for agent, hist in action_histories.items()}
                
                for agent in self.agent_info['agent_list']:
                    next_obs_histories[agent].append(next_obs_dict_partial[agent])
                    next_action_histories[agent].append(int(actions[self.agent_info['agent_list'].index(agent)]))
                    if len(next_obs_histories[agent]) > self.network_params['history_len']:
                        next_obs_histories[agent].pop(0)
                        next_action_histories[agent].pop(0)
                
                # Store transition
                self._store_transition(
                    obs_histories, action_histories, actions, reward,
                    next_obs_histories, next_action_histories, done, state, next_state, t
                )
                
                obs_histories = next_obs_histories
                action_histories = next_action_histories
                state = next_state
                episode_reward += reward
                self.global_step += 1
                t+=1
                
                # Training step
                if self.global_step % self.training_params['update_freq'] == 0:
                    self._update_networks()
                
                # Update target networks
                self._update_target_networks()
            
            self.train_rewards.append(episode_reward)
            
            # Save episode reward
            with open('qtran_groupdr_reward.txt', 'a') as f:
                f.write(f"[Episode {episode}] episode_reward: {episode_reward:.6f}\n")
            
            # Periodic evaluation
            if (episode + 1) % self.training_params['eval_freq'] == 0:
                self._evaluate_policy(episode + 1)
        
        return self._save_results()
    
    def _evaluate_policy(self, episode):
        """Evaluate current policy on all evaluation configurations."""
        eval_results = {}
        rng = np.random.default_rng(self.training_params['env_seed'])
        seeds = rng.integers(0, 200, size=self.training_params['num_eval_episodes'], dtype=int)
        seeds = seeds.tolist()
        for eval_config in EVAL_CONFIGS:
            eval_result = parallel_evaluate_policy(
                self.q_nets, eval_config, self.device,
                num_episodes=self.training_params['num_eval_episodes'],
                num_workers=min(4, self.training_params['num_eval_episodes']),
                history_len=self.network_params['history_len'],
                episode_seeds=seeds
            )
            eval_results[eval_config['name']] = eval_result
            print(f"[GroupDR] Episode {episode} Eval on {eval_config['name']}: "
                  f"Reward={eval_result['mean_reward']:.2f} ± {eval_result['std_reward']:.2f}")
        
        self.eval_checkpoints[episode] = eval_results
    
    def _save_results(self):
        """Save training results and plots."""
        results = {
            'train_rewards': self.train_rewards,
            'eval_checkpoints': self.eval_checkpoints,
        }
        
        results_dir = create_results_dir('qtran_groupdr', 0.0)
        os.makedirs(results_dir, exist_ok=True)
        
        # Save results
        with open(os.path.join(results_dir, 'results.pkl'), 'wb') as f:
            pickle.dump(results, f)
        
        # Save training curve
        plt.figure(figsize=(12, 6))
        plt.plot(self.train_rewards, label='Training Reward')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title(f'QTRAN GroupDR Training Reward')
        plt.legend()
        plt.savefig(os.path.join(results_dir, 'training_curve.png'))
        plt.close()
        
        # Save per-agent Q-network parameters
        network_param_dir = os.path.join(results_dir, 'network_parameter')
        os.makedirs(network_param_dir, exist_ok=True)
        for idx, q_net in enumerate(self.q_nets):
            torch.save(q_net.state_dict(), os.path.join(network_param_dir, f'agent_{idx}_q_network.pt'))
        
        return results 