"""
QMIX algorithm implementation with contamination robustness.
"""

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import pickle

from configs.config import TRAINING_CONFIG, EVAL_CONFIGS, get_training_params, get_network_params, create_results_dir, create_log_file
from environments.env_utils import create_environment, process_obs_dict_partial, pad_history, pad_action_history
from networks.networks import QNetwork, QMixer
from utils.replay_buffer import ReplayBuffer
from utils.evaluation import select_actions, parallel_evaluate_policy

class RobustQMIX:
    """
    Robust QMIX algorithm implementation with contamination robustness.
    """
    
    def __init__(self, rho_value, device=None):
        """
        Initialize Robust QMIX algorithm.
        
        Args:
            rho_value: Contamination parameter for robust training
            device: Device to run networks on
        """
        self.rho_value = rho_value
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Get configuration parameters
        self.training_params = get_training_params('qmix')
        self.network_params = get_network_params()
        
        # set random seed
        self._set_random_seed()  

        # rng for random seed
        self.action_rng = np.random.default_rng(self.training_params['random_seed'])
        self.buffer_rng = np.random.default_rng(self.training_params['batch_seed'])

        # Create environment
        self.env = create_environment(TRAINING_CONFIG)
        self.agent_info = self._get_agent_info()
        
        # Initialize networks
        self.q_nets, self.target_nets = self._initialize_networks()
        self.mixer, self.target_mixer = self._initialize_mixers()
        self.optimizer = self._initialize_optimizer()
        
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
    
    def _initialize_networks(self):
        """Initialize Q-networks and target networks."""
        q_nets = [
            QNetwork(obs_dim, self.agent_info['n_actions'], 
                    hidden_dim=self.network_params['hidden_dim']).to(self.device)
            for obs_dim in self.agent_info['obs_dims']
        ]
        
        target_nets = [
            QNetwork(obs_dim, self.agent_info['n_actions'], 
                    hidden_dim=self.network_params['hidden_dim']).to(self.device)
            for obs_dim in self.agent_info['obs_dims']
        ]
        
        # Initialize target networks with Q-network weights
        for q_net, target_net in zip(q_nets, target_nets):
            target_net.load_state_dict(q_net.state_dict())
            
        return q_nets, target_nets
    
    def _initialize_mixers(self):
        """Initialize QMIX mixer and target mixer."""
        mixer = QMixer(
            state_shape=(self.network_params['state_dim'],),
            mixing_embed_dim=self.network_params['mixing_embed_dim'],
            n_agents=self.agent_info['n_agents'],
            device=self.device
        )
        
        target_mixer = QMixer(
            state_shape=(self.network_params['state_dim'],),
            mixing_embed_dim=self.network_params['mixing_embed_dim'],
            n_agents=self.agent_info['n_agents'],
            device=self.device
        )
        
        target_mixer.load_state_dict(mixer.state_dict())
        return mixer, target_mixer
    
    def _initialize_optimizer(self):
        """Initialize optimizer for all networks."""
        all_params = list(self.mixer.parameters())
        for q_net in self.q_nets:
            all_params += list(q_net.parameters())
        
        return torch.optim.Adam(
            all_params, 
            lr=self.network_params['learning_rate']['mixer']
        )
    
    def _is_valid_transition(self, transition):
        """Check if a transition tuple is valid."""
        if len(transition) != 9:
            return False
        obs_histories, action_histories, actions, _, next_obs_histories, next_action_histories, _, state, next_state = transition
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
                         next_obs_histories, next_action_histories, done, state, next_state):
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
            next_state
        )
        
        if self._is_valid_transition(transition):
            self.buffer.add(transition)
    
    def _update_networks(self):
        """Update Q-networks and mixer using experience replay."""
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
        next_state_batch = torch.FloatTensor(np.stack([transition[8] for transition in batch])).to(self.device)
        
        # Compute Q-values for current observations
        all_q_values = [
            q_net(obs_batch[i], action_histories_batch[i].unsqueeze(-1))[0]
            for i, q_net in enumerate(self.q_nets)
        ]
        
        # Gather Q-values for chosen actions
        chosen_q_values = []
        for i, q_values in enumerate(all_q_values):
            if actions_batch.dim() > 1:
                agent_actions = actions_batch[:, i].unsqueeze(-1)
            else:
                agent_actions = torch.zeros(actions_batch.size(0), 1, dtype=torch.long, device=self.device)
                for j, acts in enumerate([b[2] for b in batch]):
                    agent_actions[j, 0] = acts[i]
            chosen_q = q_values.gather(1, agent_actions).squeeze(-1)
            chosen_q_values.append(chosen_q)
        
        chosen_q_stack = torch.stack(chosen_q_values, dim=1).unsqueeze(-1)
        global_q = self.mixer.mix(chosen_q_stack, state_batch).squeeze(-1)
        
        # Compute target Q-values
        with torch.no_grad():
            next_max_q_values = [
                target_net(next_obs_batch[i], next_action_histories_batch[i].unsqueeze(-1))[0].max(dim=1)[0]
                for i, target_net in enumerate(self.target_nets)
            ]
            next_max_q_stack = torch.stack(next_max_q_values, dim=1).unsqueeze(-1)
            global_next_max_q = self.target_mixer.mix(next_max_q_stack, next_state_batch).squeeze(-1)
            target = reward_batch + (1-self.rho_value)*self.network_params['gamma'] * global_next_max_q * (1 - done_batch)
        
        # Compute loss and update networks
        loss = nn.MSELoss()(global_q, target)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.optimizer.param_groups[0]['params'], 
                                     max_norm=self.network_params['grad_clip_norm'])
        self.optimizer.step()
        
        # Log training progress
        if self.global_step % 1000 == 0:
            self._log_training_progress(loss, global_q, target)
    
    def _log_training_progress(self, loss, global_q, target):
        """Log training progress to file."""
        log_file = create_log_file('qmix')
        with open(log_file, 'a') as f:
            f.write(f"[Step {self.global_step}] loss: {loss.item():.6f}\n")
            f.write(f"[Step {self.global_step}] global_q: mean={global_q.mean().item():.4f}, "
                   f"min={global_q.min().item():.4f}, max={global_q.max().item():.4f}\n")
            f.write(f"[Step {self.global_step}] target: mean={target.mean().item():.4f}, "
                   f"min={target.min().item():.4f}, max={target.max().item():.4f}\n")
    
    def _update_target_networks(self):
        """Update target networks."""
        if self.global_step % self.training_params['target_update_freq'] == 0:
            for q_net, target_net in zip(self.q_nets, self.target_nets):
                target_net.load_state_dict(q_net.state_dict())
            self.target_mixer.load_state_dict(self.mixer.state_dict())
    
    def train(self):
        """Main training loop."""
        rng = np.random.default_rng(self.training_params['env_seed'])
        seeds = rng.integers(0, 200, size=self.training_params['num_episodes'], dtype=int)
        seeds = seeds.tolist()
        pbar = tqdm(range(self.training_params['num_episodes']), 
                   desc=f'Training QMIX (rho={self.rho_value})')
        
        for episode in pbar:
            episode_seed = seeds[episode]
            obs_dict, _ = self.env.reset(episode_seed)
            obs_dict_partial = process_obs_dict_partial(obs_dict, self.env)
            
            obs_histories = {agent: [obs_dict_partial[agent]] for agent in self.agent_info['agent_list']}
            action_histories = {agent: [0] for agent in self.agent_info['agent_list']}
            
            done = False
            episode_reward = 0
            state = obs_dict[0]  # Global state
            
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
                    next_obs_histories, next_action_histories, done, state, next_state
                )
                
                obs_histories = next_obs_histories
                action_histories = next_action_histories
                state = next_state
                episode_reward += reward
                self.global_step += 1
                
                # Training step
                if self.global_step % self.training_params['update_freq'] == 0:
                    self._update_networks()
                
                # Update target networks
                self._update_target_networks()
            
            self.train_rewards.append(episode_reward)
            
            # Periodic evaluation
            if (episode + 1) % self.training_params['eval_freq'] == 0:
                self._evaluate_policy(episode + 1)
        
        return self._save_results()
    
    def _evaluate_policy(self, episode):
        """Evaluate current policy on all evaluation configurations."""
        eval_results = {}
        rng = np.random.default_rng(self.training_params['env_seed'])
        arr = rng.integers(0, 200, size=self.training_params['num_eval_episodes'], dtype=int)
        arr = arr.tolist()
        for eval_config in EVAL_CONFIGS:
            eval_result = parallel_evaluate_policy(
                self.q_nets, eval_config, self.device,
                num_episodes=self.training_params['num_eval_episodes'],
                num_workers=min(4, self.training_params['num_eval_episodes']),
                history_len=self.network_params['history_len'],
                episode_seeds=arr
            )
            eval_results[eval_config['name']] = eval_result
            print(f"[rho={self.rho_value}] Episode {episode} Eval on {eval_config['name']}: "
                  f"Reward={eval_result['mean_reward']:.2f} ± {eval_result['std_reward']:.2f}")
        
        self.eval_checkpoints[episode] = eval_results
    
    def _save_results(self):
        """Save training results and plots."""
        results = {
            'train_rewards': self.train_rewards,
            'eval_checkpoints': self.eval_checkpoints,
            'rho_value': self.rho_value
        }
        
        results_dir = create_results_dir('qmix', self.rho_value)
        os.makedirs(results_dir, exist_ok=True)
        
        # Save results
        with open(os.path.join(results_dir, 'results.pkl'), 'wb') as f:
            pickle.dump(results, f)
        
        # Save training curve
        plt.figure(figsize=(12, 6))
        plt.plot(self.train_rewards, label='Training Reward')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title(f'Robust QMIX Training Reward (rho={self.rho_value})')
        plt.legend()
        plt.savefig(os.path.join(results_dir, 'training_curve.png'))
        plt.close()
        
        # Save per-agent Q-network parameters
        network_param_dir = os.path.join(results_dir, 'network_parameter')
        os.makedirs(network_param_dir, exist_ok=True)
        for idx, q_net in enumerate(self.q_nets):
            torch.save(q_net.state_dict(), os.path.join(network_param_dir, f'agent_{idx}_q_network.pt'))
        
        return results 