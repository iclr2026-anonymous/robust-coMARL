"""
Environment utilities for Robust MARL algorithms.
Contains environment creation, action space normalization, and observation processing.
"""

import numpy as np
import gymnasium as gym
from sustaingym.envs.building import MultiAgentBuildingEnv, ParameterGenerator

def normalize_action_space(env):
    """
    Normalize the action space for the environment to ensure consistency across agents.
    """
    senv = env.single_env
    senv.Qlow = -senv.ac_map.astype(np.float32)
    senv.Qhigh = senv.ac_map.astype(np.float32)
    senv.action_space = gym.spaces.MultiDiscrete(
        (senv.Qhigh * senv.DISCRETE_LENGTH - senv.Qlow * senv.DISCRETE_LENGTH).astype(np.int64)
    )
    env.action_spaces = {agent: senv.action_space[agent] for agent in env.agents}
    return env


# TODO: add more environments.
def create_environment(config):
    """
    Create a MultiAgentBuildingEnv with the given configuration.
    
    Args:
        config: Dictionary containing environment parameters
        
    Returns:
        Normalized MultiAgentBuildingEnv
    """
    params = ParameterGenerator(
        building=config['building'],
        weather=config['weather'],
        location=config['location'],
        ac_map=config['ac_map'],
        is_continuous_action=False
    )
    env = MultiAgentBuildingEnv(params)
    env = normalize_action_space(env)
    return env

def process_obs_dict_partial(obs_dict, env, obs_dim=5):
    """
    Process the full observation dict to extract partial observations for each agent.
    Each agent sees its own i-th dimension and the last 4 dimensions.
    
    Args:
        obs_dict: Full observation dictionary from environment
        env: Environment instance
        obs_dim: Dimension of partial observations
        
    Returns:
        Dictionary with partial observations for each agent
    """
    obs_dict_partial = {}
    for i, agent in enumerate(env.agents):
        if agent in obs_dict:
            obs = obs_dict[agent].copy()
            indices = [i] + list(range(obs.shape[0] - 4, obs.shape[0]))
            obs = obs[indices]
            obs_dict_partial[agent] = obs
        else:
            obs_dict_partial[agent] = np.zeros(obs_dim, dtype=np.float32)
    return obs_dict_partial

def pad_history(history, history_len, obs_dim=5):
    """
    Pad the observation history to a fixed length with zeros if needed.
    
    Args:
        history: List of observations
        history_len: Target length
        obs_dim: Observation dimension
        
    Returns:
        Padded history list
    """
    if len(history) < history_len:
        pad = [np.zeros(obs_dim, dtype=np.float32)] * (history_len - len(history))
        return pad + history
    else:
        return history[-history_len:]

def pad_action_history(action_history, history_len):
    """
    Pad the action history to a fixed length with zeros if needed.
    
    Args:
        action_history: List of actions
        history_len: Target length
        
    Returns:
        Padded action history list
    """
    if len(action_history) < history_len:
        pad = [0] * (history_len - len(action_history))
        return pad + action_history
    else:
        return action_history[-history_len:]

def get_agent_info(env):
    """
    Get basic information about the environment and agents.
    
    Args:
        env: Environment instance
        
    Returns:
        Dictionary containing agent information
    """
    return {
        'n_agents': len(env.agents),
        'agent_list': list(env.agents),
        'obs_dims': [5 for _ in env.agents],  # Each agent gets a 5-dim input
        'n_actions': env.action_space(env.agents[0]).n if hasattr(env, 'action_space') else env.action_spaces[env.agents[0]].n
    } 