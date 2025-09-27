"""
Unified network architectures for Robust MARL algorithms.
Contains Q-Network, G-Network, QMixer, and other network components.
"""

import torch
import torch.nn as nn
import numpy as np
from configs.config import EVAL_CONFIGS

class QNetwork(nn.Module):
    """
    Q-Network for each agent. Uses an LSTM to process a sequence of partial observations and action history.
    Input: (batch, seq_len, obs_dim + n_actions) - concatenated observations and one-hot actions
    Output: Q-values for each action
    """
    def __init__(self, obs_dim, n_actions, hidden_dim=64):
        super(QNetwork, self).__init__()
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        
        # Process concatenated observations and actions
        self.fc1 = nn.Linear(obs_dim + n_actions, hidden_dim)
        self.relu1 = nn.ReLU()
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.relu2 = nn.ReLU()
        self.fc_out = nn.Linear(hidden_dim, n_actions)

    def forward(self, obs_seq, action_seq, hidden=None):
        # obs_seq: (batch, seq_len, obs_dim)
        # action_seq: (batch, seq_len, 1) - contains action indices
        batch, seq_len, obs_dim = obs_seq.shape
        
        # Convert action indices to one-hot encoding
        action_seq_flat = action_seq.view(-1, 1)  # (batch*seq_len, 1)
        action_one_hot = torch.zeros(batch * seq_len, self.n_actions, device=action_seq.device)
        action_one_hot.scatter_(1, action_seq_flat.long(), 1)  # Convert to one-hot
        action_one_hot = action_one_hot.view(batch, seq_len, self.n_actions)  # (batch, seq_len, n_actions)
        
        # Concatenate observations and one-hot actions along feature dimension
        combined_input = torch.cat([obs_seq, action_one_hot], dim=2)  # (batch, seq_len, obs_dim + n_actions)
        
        # Process through the network
        features = self.fc1(combined_input)
        features = self.relu1(features)
        lstm_out, hidden = self.lstm(features, hidden)
        out = lstm_out[:, -1, :]  # Take the last time step output
        out = self.relu2(out)
        q_values = self.fc_out(out)
        return q_values, hidden

class QNetworkWithPhi(QNetwork):
    """
    Extended Q-Network that also returns phi_t (hidden features) for QTRAN.
    """
    def forward(self, obs_seq, action_seq, hidden=None):
        batch, seq_len, obs_dim = obs_seq.shape
        
        # Convert action indices to one-hot encoding
        action_seq_flat = action_seq.view(-1, 1)
        action_one_hot = torch.zeros(batch * seq_len, self.n_actions, device=action_seq.device)
        action_one_hot.scatter_(1, action_seq_flat.long(), 1)
        action_one_hot = action_one_hot.view(batch, seq_len, self.n_actions)
        
        # Concatenate observations and one-hot actions
        combined_input = torch.cat([obs_seq, action_one_hot], dim=2)
        
        # Process through the network
        features = self.fc1(combined_input)
        features = self.relu1(features)
        lstm_out, hidden = self.lstm(features, hidden)
        phi_t = lstm_out[:, -1, :]  # Last time step output
        out = self.relu2(phi_t)
        q_values = self.fc_out(out)
        return q_values, phi_t, hidden

class GNetwork(nn.Module):
    """
    G-Network that takes the current state and joint action as input, outputs a g value.
    This network learns to approximate the joint action-value function.
    """
    def __init__(self, state_dim, n_agents, n_actions, hidden_dim=64):
        super(GNetwork, self).__init__()
        self.state_dim = state_dim
        self.n_agents = n_agents
        self.n_actions = n_actions
        
        # Joint action input: concatenated one-hot actions from all agents
        joint_action_dim = n_agents * n_actions
        
        # State processing
        self.state_fc = nn.Linear(state_dim, hidden_dim)
        self.state_relu = nn.ReLU()
        
        # Joint action processing
        self.joint_action_fc = nn.Linear(joint_action_dim, hidden_dim)
        self.joint_action_relu = nn.ReLU()
        
        # Combined processing
        self.combined_fc = nn.Linear(hidden_dim * 2, hidden_dim)
        self.combined_relu = nn.ReLU()
        
        # Output layer
        self.output_fc = nn.Linear(hidden_dim, 1)

    def forward(self, state, joint_action):
        # state: (batch, state_dim)
        # joint_action: (batch, n_agents) - contains action indices for each agent
        
        batch_size = joint_action.shape[0]
        
        # Process state
        state_features = self.state_relu(self.state_fc(state))
        
        # Convert joint_action to one-hot encoding
        joint_action_one_hot_list = []
        for i in range(self.n_agents):
            agent_actions = joint_action[:, i].long()
            agent_one_hot = torch.zeros(batch_size, self.n_actions, device=joint_action.device)
            agent_actions = torch.clamp(agent_actions, 0, self.n_actions - 1)
            agent_one_hot.scatter_(1, agent_actions.unsqueeze(1), 1)
            joint_action_one_hot_list.append(agent_one_hot)
        
        joint_action_one_hot = torch.cat(joint_action_one_hot_list, dim=1)  # (batch, n_agents * n_actions)
        joint_action_features = self.joint_action_relu(self.joint_action_fc(joint_action_one_hot.float()))
        
        # Combine state and joint action features
        combined_features = torch.cat([state_features, joint_action_features], dim=1)
        combined_features = self.combined_relu(self.combined_fc(combined_features))
        
        # Output g value
        g_value = self.output_fc(combined_features)
        return g_value

class GNetworkWithHidden(GNetwork):
    """
    Extended G-Network that also processes hidden features from individual Q-networks.
    """
    def __init__(self, state_dim, n_agents, n_actions, hidden_dim=64):
        super().__init__(state_dim, n_agents, n_actions, hidden_dim)
        
        # Hidden features processing
        self.hidden_fc = nn.Linear(hidden_dim, hidden_dim)
        self.hidden_relu = nn.ReLU()
        
        # Update combined processing for 3 inputs
        self.combined_fc = nn.Linear(hidden_dim * 3, hidden_dim)

    def forward(self, state, joint_action, hidden_features=None):
        batch_size = joint_action.shape[0]
        
        # Process state
        state_features = self.state_relu(self.state_fc(state))
        
        # Convert joint_action to one-hot encoding
        joint_action_one_hot_list = []
        for i in range(self.n_agents):
            agent_actions = joint_action[:, i].long()
            agent_one_hot = torch.zeros(batch_size, self.n_actions, device=joint_action.device)
            agent_actions = torch.clamp(agent_actions, 0, self.n_actions - 1)
            agent_one_hot.scatter_(1, agent_actions.unsqueeze(1), 1)
            joint_action_one_hot_list.append(agent_one_hot)
        
        joint_action_one_hot = torch.cat(joint_action_one_hot_list, dim=1)
        joint_action_features = self.joint_action_relu(self.joint_action_fc(joint_action_one_hot.float()))
        
        # Process hidden features if available
        if hidden_features is not None:
            if isinstance(hidden_features, list):
                hidden_states = [h[0].detach() if isinstance(h, tuple) else h for h in hidden_features]
                hidden_features = torch.stack(hidden_states, dim=1)
            hidden_features = torch.sum(hidden_features, dim=1)
            hidden_features = self.hidden_relu(self.hidden_fc(hidden_features))
        else:
            hidden_features = torch.zeros(batch_size, self.hidden_fc.out_features, device=state.device)
        
        # Combine all features
        combined_features = torch.cat([state_features, joint_action_features, hidden_features], dim=1)
        combined_features = self.combined_relu(self.combined_fc(combined_features))
        
        # Output g value
        g_value = self.output_fc(combined_features)
        return g_value

class QMixer(nn.Module):
    """
    QMIX mixing network that combines individual Q-values using hypernetworks.
    """
    def __init__(self, state_shape, mixing_embed_dim, n_agents, device):
        super(QMixer, self).__init__()
        self.embed_dim = mixing_embed_dim
        self.state_dim = int(np.prod(state_shape))
        self.n_agents = n_agents
        self.device = device
        
        # Hypernetworks
        self.hyper_w_1 = nn.Linear(self.state_dim, n_agents * self.embed_dim)
        self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)
        
        # Initialize with smaller weights
        nn.init.xavier_uniform_(self.hyper_w_1.weight, gain=0.1)
        nn.init.constant_(self.hyper_w_1.bias, 0.0)
        nn.init.xavier_uniform_(self.hyper_w_final.weight, gain=0.1)
        nn.init.constant_(self.hyper_w_final.bias, 0.0)
        nn.init.xavier_uniform_(self.hyper_b_1.weight, gain=0.1)
        nn.init.constant_(self.hyper_b_1.bias, 0.0)
        
        # State-dependent bias network
        self.V = nn.Sequential(
            nn.Linear(self.state_dim, self.embed_dim, device=self.device),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 1, device=self.device),
        )
        
        # Initialize V network with smaller weights
        for layer in self.V:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.1)
                nn.init.constant_(layer.bias, 0.0)

    def mix(self, chosen_action_value: torch.Tensor, state: torch.Tensor):
        bs = chosen_action_value.shape[:-2]
        state = state.view(-1, self.state_dim)
        chosen_action_value = chosen_action_value.view(-1, 1, self.n_agents)
        
        # First layer
        w1 = torch.abs(self.hyper_w_1(state))
        b1 = self.hyper_b_1(state)
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        hidden = nn.functional.elu(
            torch.bmm(chosen_action_value, w1) + b1
        )
        
        # Second layer
        w_final = torch.abs(self.hyper_w_final(state))
        w_final = w_final.view(-1, self.embed_dim, 1)
        
        # State-dependent bias
        v = self.V(state).view(-1, 1, 1)
        
        # Compute final output
        y = torch.bmm(hidden, w_final) + v
        q_tot = y.view(*bs, 1)
        return q_tot

class JointActionValueNetwork(nn.Module):
    """
    Joint action-value network for QTRAN architecture.
    """
    def __init__(self, n_agents, n_actions, hidden_dim=32):
        super(JointActionValueNetwork, self).__init__()
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim

        # Joint action feature processing
        self.joint_action_fc = nn.Linear(n_agents * n_actions, hidden_dim)
        self.joint_action_relu = nn.ReLU()

        # Combined processing
        self.combined_fc = nn.Linear(hidden_dim * 2, hidden_dim)
        self.combined_relu = nn.ReLU()
        self.out_fc = nn.Linear(hidden_dim, 1)

    def forward(self, joint_action, cached_hidden_states=None):
        # Use cached hidden features if available
        hidden_features = cached_hidden_states
        
        if hidden_features is not None:
            if isinstance(hidden_features, list):
                hidden_states = [h[0].detach() if isinstance(h, tuple) else h for h in hidden_features]
                hidden_features = torch.stack(hidden_states, dim=1)
            hidden_features = torch.sum(hidden_features, dim=1)
            hidden_features = hidden_features.squeeze(0)

        # joint_action shape: (batch_size, n_agents)
        batch_size = joint_action.shape[0]
        
        # Create one-hot encoding for each agent's action
        joint_action_one_hot_list = []
        for i in range(self.n_agents):
            agent_actions = joint_action[:, i].long()
            agent_one_hot = torch.zeros(batch_size, self.n_actions, device=joint_action.device)
            agent_actions = torch.clamp(agent_actions, 0, self.n_actions - 1)
            agent_one_hot.scatter_(1, agent_actions.unsqueeze(1), 1)
            joint_action_one_hot_list.append(agent_one_hot)
        
        # Concatenate all agent one-hot encodings
        joint_action_one_hot = torch.cat(joint_action_one_hot_list, dim=1)
        joint_action_features = self.joint_action_relu(self.joint_action_fc(joint_action_one_hot.float()))

        # Combine and output
        if hidden_features is None:
            combined = joint_action_features
        else:
            combined = torch.cat([joint_action_features, hidden_features], dim=1)
        
        combined = self.combined_relu(self.combined_fc(combined))
        joint_q_value = self.out_fc(combined)
        return joint_q_value, hidden_features

class StateValueNetwork(nn.Module):
    """
    State-value network for QTRAN architecture.
    """
    def __init__(self, state_dim, n_agents, hidden_dim=32):
        super(StateValueNetwork, self).__init__()
        self.state_dim = state_dim
        self.n_agents = n_agents
        self.hidden_dim = hidden_dim
        
        # State processing layers
        self.state_fc = nn.Linear(state_dim, hidden_dim)
        self.state_relu = nn.ReLU()
        
        # Hidden features processing
        self.hidden_fc = nn.Linear(hidden_dim, hidden_dim)
        self.hidden_relu = nn.ReLU()
        
        # Combined processing layers
        self.combined_fc = nn.Linear(hidden_dim * 2, hidden_dim)
        self.combined_relu = nn.ReLU()
        self.out_fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, current_state, cached_hidden_states=None):
        batch_size = current_state.shape[0]
        current_state = current_state.view(batch_size, -1)
        state_features = self.state_relu(self.state_fc(current_state))
        
        if cached_hidden_states is not None:
            if isinstance(cached_hidden_states, list):
                hidden_states = [h[0].detach() if isinstance(h, tuple) else h for h in cached_hidden_states]
                hidden_features = torch.stack(hidden_states, dim=1)
            hidden_features = torch.sum(hidden_features, dim=1)
            hidden_features = self.hidden_relu(self.hidden_fc(hidden_features))
        else:
            hidden_features = torch.zeros(batch_size, self.hidden_dim, device=current_state.device)
        
        # Combine state features and hidden features
        combined_features = torch.cat([state_features, hidden_features], dim=1)
        
        # Final processing for state value
        combined = self.combined_relu(self.combined_fc(combined_features))
        state_value = self.out_fc(combined)
        
        return state_value


class EnvEstimator(nn.Module):
    """
    Environment reward estimator.
    Inputs:
      - state: (batch, state_dim)
      - joint_action: (batch, n_agents) with action indices per agent
    Output:
      - rewards_pred: (batch, num_envs) where num_envs == len(EVAL_CONFIGS)
        Each column corresponds to the estimated average immediate reward
        under a specific environment for the given state and joint action.
    """
    def __init__(self, state_dim: int, n_agents: int, n_actions: int, hidden_dim: int = 64):
        super().__init__()
        self.state_dim = state_dim
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.num_envs = len(EVAL_CONFIGS)
        
        # State processing
        self.state_fc = nn.Linear(state_dim, hidden_dim)
        self.state_relu = nn.ReLU()
        
        # Joint action processing (one-hot per agent, then concat)
        self.joint_action_fc = nn.Linear(n_agents * n_actions, hidden_dim)
        self.joint_action_relu = nn.ReLU()
        
        # Combine and predict per-environment reward
        self.combined_fc = nn.Linear(hidden_dim * 2, hidden_dim)
        self.combined_relu = nn.ReLU()
        self.out_fc = nn.Linear(hidden_dim, self.num_envs)
        
    def forward(self, state: torch.Tensor, joint_action: torch.Tensor) -> torch.Tensor:
        batch_size = joint_action.shape[0]
        
        # State path
        state_features = self.state_relu(self.state_fc(state))
        
        # Joint action one-hot encoding
        joint_action_one_hot_list = []
        for i in range(self.n_agents):
            agent_actions = joint_action[:, i].long()
            agent_one_hot = torch.zeros(batch_size, self.n_actions, device=joint_action.device)
            agent_actions = torch.clamp(agent_actions, 0, self.n_actions - 1)
            agent_one_hot.scatter_(1, agent_actions.unsqueeze(1), 1)
            joint_action_one_hot_list.append(agent_one_hot)
        joint_action_one_hot = torch.cat(joint_action_one_hot_list, dim=1)
        joint_action_features = self.joint_action_relu(self.joint_action_fc(joint_action_one_hot.float()))
        
        # Combine
        combined = torch.cat([state_features, joint_action_features], dim=1)
        combined = self.combined_relu(self.combined_fc(combined))
        rewards_pred = self.out_fc(combined)
        return rewards_pred 


class EnvEstimatorWithStep(EnvEstimator):
    """
    EnvEstimator that additionally takes the timestep in an episode as input.
    Inputs:
      - state: (batch, state_dim)
      - joint_action: (batch, n_agents) with action indices per agent
      - timestep: (batch,) or (batch, 1) numeric step index in the episode
    Output:
      - rewards_pred: (batch, num_envs)
    """
    def __init__(self, state_dim: int, n_agents: int, n_actions: int, hidden_dim: int = 64):
        super().__init__(state_dim, n_agents, n_actions, hidden_dim)
        # Additional processing for timestep
        self.step_fc = nn.Linear(1, hidden_dim)
        self.step_relu = nn.ReLU()
        # Override combined_fc to accept 3 inputs (state, joint_action, timestep)
        self.combined_fc = nn.Linear(hidden_dim * 3, hidden_dim)

    def forward(self, state: torch.Tensor, joint_action: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        batch_size = joint_action.shape[0]

        # State path
        state_features = self.state_relu(self.state_fc(state))

        # Joint action one-hot encoding
        joint_action_one_hot_list = []
        for i in range(self.n_agents):
            agent_actions = joint_action[:, i].long()
            agent_one_hot = torch.zeros(batch_size, self.n_actions, device=joint_action.device)
            agent_actions = torch.clamp(agent_actions, 0, self.n_actions - 1)
            agent_one_hot.scatter_(1, agent_actions.unsqueeze(1), 1)
            joint_action_one_hot_list.append(agent_one_hot)
        joint_action_one_hot = torch.cat(joint_action_one_hot_list, dim=1)
        joint_action_features = self.joint_action_relu(self.joint_action_fc(joint_action_one_hot.float()))

        # Timestep path (ensure shape (batch, 1))
        if timestep.dim() == 1:
            timestep = timestep.view(-1, 1)
        step_features = self.step_relu(self.step_fc(timestep.float().to(state.device)))

        # Combine
        combined = torch.cat([state_features, joint_action_features, step_features], dim=1)
        combined = self.combined_relu(self.combined_fc(combined))
        rewards_pred = self.out_fc(combined)
        return rewards_pred 