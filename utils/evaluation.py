"""
Evaluation utilities for Robust MARL algorithms.
Contains policy evaluation and parallel evaluation functionality.
"""

import torch
import numpy as np
import multiprocessing as mp
from sustaingym.envs.building import MultiAgentBuildingEnv, ParameterGenerator
from environments.env_utils import normalize_action_space, process_obs_dict_partial
from networks.networks import QNetwork, QNetworkWithPhi


def select_actions(q_nets, obs_histories, action_histories, agent_list, device, history_len=8):
    """
    Select actions for all agents using their Q-networks and observation/action histories.
    Uses greedy action selection (argmax Q-value).
    
    Args:
        q_nets: List of Q-networks for each agent
        obs_histories: Dictionary of observation histories for each agent
        action_histories: Dictionary of action histories for each agent
        agent_list: List of agent names
        device: Device to run networks on
        history_len: Length of history to use
        
    Returns:
        List of selected actions for each agent
    """
    actions = []
    for i, agent in enumerate(agent_list):
        q_net = q_nets[i]
        obs_history = obs_histories[agent]
        action_history = action_histories[agent]
        
        # Pad observation history
        if len(obs_history) < history_len:
            pad = [np.zeros_like(obs_history[0])] * (history_len - len(obs_history)) if obs_history else [np.zeros(5)] * history_len
            obs_history = pad + obs_history
        else:
            obs_history = obs_history[-history_len:]
            
        # Pad action history
        if len(action_history) < history_len:
            pad = [0] * (history_len - len(action_history)) if action_history else [0] * history_len
            action_history = pad + action_history
        else:
            action_history = action_history[-history_len:]
            
        obs_seq = torch.FloatTensor(np.stack(obs_history)).unsqueeze(0).to(device)
        action_seq = torch.FloatTensor(action_history).unsqueeze(0).unsqueeze(-1).to(device)
        
        # Handle different Q-network types
        if hasattr(q_net, 'forward') and q_net.__class__.__name__ == 'QNetworkWithPhi':
            q_values, _, _ = q_net(obs_seq, action_seq)
        else:
            q_values, _ = q_net(obs_seq, action_seq)
            
        action = q_values.argmax(dim=-1).item()
        actions.append(action)
    return actions

def evaluate_policy(q_nets, env, device, num_episodes=10, history_len=8):
    """
    Evaluate the current policy (Q-networks) in the given environment.
    Returns mean and std of rewards and episode lengths.
    
    Args:
        q_nets: List of Q-networks for each agent
        env: Environment to evaluate in
        device: Device to run networks on
        num_episodes: Number of episodes to evaluate
        history_len: Length of history to use
        
    Returns:
        Dictionary with evaluation statistics
    """
    episode_rewards = []
    episode_lengths = []
    
    for _ in range(num_episodes):
        obs_dict, _ = env.reset()
        obs_dict_partial = process_obs_dict_partial(obs_dict, env)
        agent_list = list(env.agents)
        
        # Initialize observation and action histories for each agent
        obs_histories = {agent: [obs_dict_partial[agent]] for agent in agent_list}
        action_histories = {agent: [0] for agent in agent_list}
        
        done = False
        episode_reward = 0
        episode_length = 0
        
        while not done:
            actions = select_actions(q_nets, obs_histories, action_histories, agent_list, device, history_len)
            
            action_dict = {agent: act for agent, act in zip(agent_list, actions)}
            next_obs_dict, rewards, dones, _, _ = env.step(action_dict)
            
            episode_reward += sum(rewards.values())
            done = any(dones.values())
            episode_length += 1
            
            if done:
                break
                
            next_obs_dict_partial = process_obs_dict_partial(next_obs_dict, env)
            
            # Update observation and action histories for each agent
            for agent in agent_list:
                obs_histories[agent].append(next_obs_dict_partial[agent])
                action_histories[agent].append(int(actions[agent_list.index(agent)]))
                if len(obs_histories[agent]) > history_len:
                    obs_histories[agent].pop(0)
                    action_histories[agent].pop(0)
                    
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
    return {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths)
    }

def _single_eval_worker(args):
    """
    Worker for a single evaluation episode. Used for parallel evaluation.
    
    Args:
        args: Tuple containing (q_net_state_dicts, eval_config, device_str, history_len)
        
    Returns:
        Tuple of (episode_reward, episode_length)
    """
    q_net_state_dicts, eval_config, device_str, history_len, task_seed = args
    # Re-create environment and Q-nets in subprocess
    env = MultiAgentBuildingEnv(
        ParameterGenerator(
            building=eval_config['building'],
            weather=eval_config['weather'],
            location=eval_config['location'],
            ac_map=eval_config['ac_map'],
            is_continuous_action=False
        )
    )
    env = normalize_action_space(env)
    
    device = torch.device(device_str)
    obs_dims = [5 for _ in env.agents]
    n_actions = env.action_spaces[env.agents[0]].n
    
    # Create Q-networks (assume standard QNetwork for evaluation)
    q_nets = [QNetwork(obs_dim, n_actions).to(device) for obs_dim in obs_dims]
    
    # Load state dicts
    for i, q_net in enumerate(q_nets):
        q_net.load_state_dict(q_net_state_dicts[i])
    
    # Run one episode
    obs_dict, _ = env.reset(task_seed)
    obs_dict_partial = process_obs_dict_partial(obs_dict, env)
    agent_list = list(env.agents)
    
    # Initialize observation and action histories for each agent
    obs_histories = {agent: [obs_dict_partial[agent]] for agent in agent_list}
    action_histories = {agent: [0] for agent in agent_list}
    
    done = False
    episode_reward = 0
    episode_length = 0
    
    while not done:
        actions = select_actions(q_nets, obs_histories, action_histories, agent_list, device, history_len)
        
        action_dict = {agent: act for agent, act in zip(agent_list, actions)}
        next_obs_dict, rewards, dones, _, _ = env.step(action_dict)
        
        episode_reward += sum(rewards.values())
        done = any(dones.values())
        episode_length += 1
        
        if done:
            break
            
        next_obs_dict_partial = process_obs_dict_partial(next_obs_dict, env)
        
        # Update observation and action histories for each agent
        for agent in agent_list:
            obs_histories[agent].append(next_obs_dict_partial[agent])
            action_histories[agent].append(int(actions[agent_list.index(agent)]))
            if len(obs_histories[agent]) > history_len:
                obs_histories[agent].pop(0)
                action_histories[agent].pop(0)
    
    return episode_reward, episode_length

def parallel_evaluate_policy(q_nets, eval_config, device, num_episodes=10, num_workers=10, history_len=8, episode_seeds=None):
    """
    Evaluate policy in parallel using multiprocessing for a given eval_config.
    Returns mean and std of rewards and episode lengths.
    
    Args:
        q_nets: List of Q-networks for each agent
        eval_config: Environment configuration for evaluation
        device: Device to run networks on
        num_episodes: Number of episodes to evaluate
        num_workers: Number of parallel workers
        history_len: Length of history to use
        
    Returns:
        Dictionary with evaluation statistics
    """
    # Share Q-net state dicts to subprocesses
    q_net_state_dicts = [q_net.state_dict() for q_net in q_nets]
    device_str = str(device)

    args = [(q_net_state_dicts, eval_config, device_str, history_len,episode_seeds[episode]) for episode in range(num_episodes)]
    
    with mp.get_context("spawn").Pool(processes=num_workers) as pool:
        results = pool.map(_single_eval_worker, args)
    
    episode_rewards, episode_lengths = zip(*results)
    
    return {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths)
    } 