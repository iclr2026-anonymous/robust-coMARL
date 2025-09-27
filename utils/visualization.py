"""
Visualization utilities for Robust MARL algorithms.
Focuses on plotting reward vs config at different checkpoints with different rho values.
"""

import os
import pickle
import glob
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from pathlib import Path


class RobustMARLVisualizer:
    """
    Visualization class for Robust MARL algorithms.
    Specializes in plotting reward vs config at different checkpoints with different rho values.
    """
    
    def __init__(self, results_dir: str = "results"):
        """
        Initialize the visualizer.
        
        Args:
            results_dir: Directory containing result files
        """
        self.results_dir = Path(results_dir)
        self.result_files = {}
        self.eval_configs = []
        self.rho_values = []
        self.algorithm = ""
        
    def load_results(self, algorithm: str, rho_values: Optional[List[float]] = None):
        """
        Load result files for a specific algorithm.
        Recursively searches all subdirectories for result files.
        Expects directory structure: /results/robust_{algorithm}_rho_{rho_value}_{timestamp}/results.pkl
        
        Args:
            algorithm: Algorithm name (e.g., 'vdn', 'qmix', 'qtran')
            rho_values: List of rho values to load (if None, auto-detect)
        """
        self.algorithm = algorithm
        # Find all subdirectories that match the algorithm pattern
        # Handle algorithms with underscores like 'qtran_g', 'vdn_g', etc.
        algorithm_pattern = f"robust_{algorithm}_rho_"
        subdirs = [d for d in self.results_dir.iterdir() if d.is_dir() and d.name.startswith(algorithm_pattern)]
        
        if not subdirs:
            raise FileNotFoundError(f"No result directories found for {algorithm} in {self.results_dir}")
        
        # Extract rho values from directory names and find results.pkl files
        detected_rho_values = []
        for subdir in subdirs:
            try:
                # Parse directory name: robust_qtran_g_rho_0.0_20250827_235600
                # Remove the 'robust_' prefix and algorithm name to get the rest
                dir_name = subdir.name
                if dir_name.startswith(f"robust_{algorithm}_rho_"):
                    # Extract the part after "robust_{algorithm}_rho_"
                    remaining = dir_name[len(f"robust_{algorithm}_rho_"):]
                    # The first part should be the rho value
                    rho_str = remaining.split('_')[0]
                    rho = float(rho_str)
                    
                    # Look for results.pkl file in the subdirectory
                    results_file = subdir / "results.pkl"
                    if results_file.exists():
                        detected_rho_values.append(rho)
                        self.result_files[rho] = results_file
                        print(f"Found: {subdir.name}/results.pkl (ρ={rho})")
                    else:
                        print(f"Warning: No results.pkl found in {subdir.name}")
                        
            except Exception as e:
                print(f"Warning: Could not parse rho from directory {subdir.name}: {e}")
        
        if not detected_rho_values:
            raise FileNotFoundError(f"No valid result files found for {algorithm} in {self.results_dir}")
        
        # Use provided rho values or detected ones
        if rho_values is not None:
            self.rho_values = sorted(rho_values)
            # Filter files to only include specified rho values
            self.result_files = {rho: self.result_files[rho] for rho in self.rho_values 
                               if rho in self.result_files}
        else:
            self.rho_values = sorted(detected_rho_values)
        
        print(f"Loaded {len(self.result_files)} result files for {algorithm}")
        print(f"Rho values: {self.rho_values}")
        print(f"Search directory: {self.results_dir}")
        
    def set_eval_configs(self, eval_configs: List[Dict[str, str]]):
        """
        Set the evaluation configurations.
        
        Args:
            eval_configs: List of config dictionaries with 'name' key
        """
        self.eval_configs = eval_configs
        print(f"Set {len(eval_configs)} evaluation configs")
        
    def extract_checkpoint_data(self) -> Dict[int, Dict[str, Dict[str, float]]]:
        """
        Extract checkpoint data from all loaded result files.
        
        Returns:
            Dictionary: {checkpoint: {config_name: {rho: reward}}}
        """
        checkpoint_data = {}
        
        for rho, file_path in self.result_files.items():
            try:
                with open(file_path, 'rb') as f:
                    results = pickle.load(f)
                
                eval_checkpoints = results.get('eval_checkpoints', {})
                
                for ckpt, eval_results in eval_checkpoints.items():
                    if ckpt not in checkpoint_data:
                        checkpoint_data[ckpt] = {}
                    
                    for config in self.eval_configs:
                        config_name = config['name']
                        if config_name in eval_results:
                            if config_name not in checkpoint_data[ckpt]:
                                checkpoint_data[ckpt][config_name] = {}
                            
                            checkpoint_data[ckpt][config_name][rho] = eval_results[config_name]['mean_reward']
                            
            except Exception as e:
                print(f"Warning: Could not load {file_path}: {e}")
        
        return checkpoint_data
    
    def load_baseline_results(self, baseline_algorithms: List[str] = None):
        """
        Load baseline algorithm results for comparison.
        
        Args:
            baseline_algorithms: List of baseline algorithm names (e.g., ['vdn_groupdr', 'qmix_groupdr', 'qtran_groupdr'])
        """
        if baseline_algorithms is None:
            baseline_algorithms = ['vdn_groupdr', 'qmix_groupdr', 'qtran_groupdr', 'vdn_dr']
        
        self.baseline_results = {}
        
        for baseline_alg in baseline_algorithms:
            try:
                # Find baseline result directories
                baseline_pattern = f"robust_{baseline_alg}_rho_"
                baseline_dirs = [d for d in self.results_dir.iterdir() 
                               if d.is_dir() and d.name.startswith(baseline_pattern)]
                
                if baseline_dirs:
                    # Use the first available baseline directory (usually rho=0.0)
                    baseline_dir = baseline_dirs[0]
                    baseline_file = baseline_dir / "results.pkl"
                    
                    if baseline_file.exists():
                        with open(baseline_file, 'rb') as f:
                            baseline_data = pickle.load(f)
                        self.baseline_results[baseline_alg] = baseline_data
                        print(f"Loaded baseline: {baseline_alg} from {baseline_dir.name}")
                    else:
                        print(f"Warning: No results.pkl found in baseline {baseline_dir.name}")
                else:
                    print(f"Warning: No baseline directories found for {baseline_alg}")
                    
            except Exception as e:
                print(f"Warning: Could not load baseline {baseline_alg}: {e}")
        
        print(f"Loaded {len(self.baseline_results)} baseline algorithms")

    def plot_reward_vs_config_at_checkpoint(self, 
                                          checkpoint: int, 
                                          checkpoint_data: Dict,
                                          save_dir: Optional[str] = None,
                                          figsize: Tuple[int, int] = (12, 8),
                                          show_confidence_intervals: bool = True,
                                          confidence_level: float = 0.95,
                                          show_baselines: bool = True):
        """
        Plot reward vs config at a specific checkpoint with different rho curves and baselines.
        
        Args:
            checkpoint: Checkpoint number to plot
            checkpoint_data: Data extracted from result files
            save_dir: Directory to save the plot (if None, use results_dir)
            figsize: Figure size (width, height)
            show_confidence_intervals: Whether to show confidence intervals
            confidence_level: Confidence level for intervals (0.95 = 95%)
            show_baselines: Whether to show baseline algorithm results
        """
        if checkpoint not in checkpoint_data:
            print(f"Warning: Checkpoint {checkpoint} not found in data")
            return
        
        # Get config names in order
        config_names = [cfg['name'] for cfg in self.eval_configs]
        
        # Create figure
        plt.figure(figsize=figsize)
        
        # Plot baselines first (so they appear behind main curves)
        if show_baselines and hasattr(self, 'baseline_results'):
            baseline_colors = ['gray', 'lightcoral', 'lightblue']
            baseline_styles = ['--', ':', '-.']
            
            for i, (baseline_alg, baseline_data) in enumerate(self.baseline_results.items()):
                try:
                    eval_checkpoints = baseline_data.get('eval_checkpoints', {})
                    if checkpoint in eval_checkpoints:
                        baseline_rewards = []
                        valid_configs = []
                        
                        for config_name in config_names:
                            if config_name in eval_checkpoints[checkpoint]:
                                reward = eval_checkpoints[checkpoint][config_name]['mean_reward']
                                baseline_rewards.append(reward)
                                valid_configs.append(config_name)
                        
                        if len(baseline_rewards) > 0:
                            color = baseline_colors[i % len(baseline_colors)]
                            style = baseline_styles[i % len(baseline_styles)]
                            plt.plot(range(len(valid_configs)), baseline_rewards,
                                   color=color, linestyle=style, linewidth=3, 
                                   marker='s', markersize=6, alpha=0.8,
                                   label=f'{baseline_alg} (baseline)')
                except Exception as e:
                    print(f"Warning: Could not plot baseline {baseline_alg}: {e}")
        
        # Plot each rho value as a separate curve
        for rho in self.rho_values:
            rewards = []
            errors = []
            valid_configs = []
            
            for config_name in config_names:
                if (config_name in checkpoint_data[checkpoint] and 
                    rho in checkpoint_data[checkpoint][config_name]):
                    
                    # Get reward and error (if available)
                    reward = checkpoint_data[checkpoint][config_name][rho]
                    
                    # Try to get error/std if available in original data
                    error = 0.0  # Default error
                    try:
                        # Load original file to get error
                        for file_path in self.result_files.values():
                            with open(file_path, 'rb') as f:
                                results = pickle.load(f)
                            eval_checkpoints = results.get('eval_checkpoints', {})
                            if (checkpoint in eval_checkpoints and 
                                config_name in eval_checkpoints[checkpoint]):
                                error = eval_checkpoints[checkpoint][config_name].get('std_reward', 0.0)
                                break
                    except:
                        pass
                    
                    rewards.append(reward)
                    errors.append(error)
                    valid_configs.append(config_name)
            
            if len(rewards) > 0:
                # Convert std to confidence interval
                if confidence_level == 0.95:
                    multiplier = 1.96
                elif confidence_level == 0.99:
                    multiplier = 2.58
                else:
                    multiplier = 1.96  # Default to 95%
                
                ci_values = [multiplier * err for err in errors]
                
                # Plot confidence intervals if requested
                if show_confidence_intervals and any(err > 0 for err in errors):
                    plt.fill_between(range(len(valid_configs)), 
                                   [r - ci for r, ci in zip(rewards, ci_values)],
                                   [r + ci for r, ci in zip(rewards, ci_values)],
                                   alpha=0.3, label=f'ρ={rho} ({int(confidence_level*100)}% CI)')
                
                # Plot the main curve
                plt.plot(range(len(valid_configs)), rewards, 
                        marker='o', linewidth=2, markersize=8, 
                        label=f'ρ={rho}')
        
        # Customize plot
        plt.xlabel('Configuration', fontsize=12)
        plt.ylabel(f'Mean Episode Reward (Checkpoint {checkpoint})', fontsize=12)
        # add algorithm name to the title
        title = f'Final Performances vs Config\n{self.algorithm}'
        plt.title(title, fontsize=14, fontweight='bold')
        
        # Set x-axis ticks
        plt.xticks(range(len(config_names)), config_names, rotation=45, ha='right')
        
        # Add grid and legend
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot
        if save_dir is None:
            save_dir = self.results_dir
        
        save_path = Path(save_dir) / f'reward_vs_config_ckpt{checkpoint}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
        
        plt.show()
        
    def plot_all_checkpoints(self, 
                            save_dir: Optional[str] = None,
                            figsize: Tuple[int, int] = (12, 8),
                            show_confidence_intervals: bool = True):
        """
        Plot reward vs config for all available checkpoints.
        
        Args:
            save_dir: Directory to save plots
            figsize: Figure size for each plot
            show_confidence_intervals: Whether to show confidence intervals
        """
        checkpoint_data = self.extract_checkpoint_data()
        
        if not checkpoint_data:
            print("No checkpoint data found")
            return
        
        checkpoints = sorted(checkpoint_data.keys())
        print(f"Plotting {len(checkpoints)} checkpoints: {checkpoints}")
        
        for checkpoint in checkpoints:
            self.plot_reward_vs_config_at_checkpoint(
                checkpoint, checkpoint_data, save_dir, figsize, show_confidence_intervals
            )
    
    def create_summary_plot(self, 
                           checkpoint: int,
                           checkpoint_data: Dict,
                           save_dir: Optional[str] = None,
                           figsize: Tuple[int, int] = (14, 10)):
        """
        Create a comprehensive summary plot with multiple subplots.
        
        Args:
            checkpoint: Checkpoint number
            checkpoint_data: Data extracted from result files
            save_dir: Directory to save the plot
            figsize: Figure size
        """
        if checkpoint not in checkpoint_data:
            print(f"Warning: Checkpoint {checkpoint} not found in data")
            return
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        # Subplot 1: Reward vs Config (main plot)
        config_names = [cfg['name'] for cfg in self.eval_configs]
        for rho in self.rho_values:
            rewards = []
            valid_configs = []
            for config_name in config_names:
                if (config_name in checkpoint_data[checkpoint] and 
                    rho in checkpoint_data[checkpoint][config_name]):
                    rewards.append(checkpoint_data[checkpoint][config_name][rho])
                    valid_configs.append(config_name)
            
            if len(rewards) > 0:
                ax1.plot(range(len(valid_configs)), rewards, 
                        marker='o', linewidth=2, markersize=8, 
                        label=f'ρ={rho}')
        
        ax1.set_xlabel('Configuration')
        ax1.set_ylabel('Mean Episode Reward')
        ax1.set_title(f'Reward vs Config (Checkpoint {checkpoint})')
        ax1.set_xticks(range(len(config_names)))
        ax1.set_xticklabels(config_names, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Subplot 2: Heatmap of rewards
        reward_matrix = np.full((len(self.rho_values), len(config_names)), np.nan)
        for i, rho in enumerate(self.rho_values):
            for j, config_name in enumerate(config_names):
                if (config_name in checkpoint_data[checkpoint] and 
                    rho in checkpoint_data[checkpoint][config_name]):
                    reward_matrix[i, j] = checkpoint_data[checkpoint][config_name][rho]
        
        im = ax2.imshow(reward_matrix, cmap='viridis', aspect='auto')
        ax2.set_xlabel('Configuration')
        ax2.set_ylabel('Rho Value')
        ax2.set_title('Reward Heatmap')
        ax2.set_xticks(range(len(config_names)))
        ax2.set_xticklabels(config_names, rotation=45, ha='right')
        ax2.set_yticks(range(len(self.rho_values)))
        ax2.set_yticklabels([f'{rho:.4f}' for rho in self.rho_values])
        plt.colorbar(im, ax=ax2, label='Reward')
        
        # Subplot 3: Rho vs Reward for each config
        for config_name in config_names:
            rhos = []
            rewards = []
            for rho in self.rho_values:
                if (config_name in checkpoint_data[checkpoint] and 
                    rho in checkpoint_data[checkpoint][config_name]):
                    rhos.append(rho)
                    rewards.append(checkpoint_data[checkpoint][config_name][rho])
            
            if len(rewards) > 0:
                ax3.plot(rhos, rewards, marker='o', linewidth=2, 
                        markersize=6, label=config_name)
        
        ax3.set_xlabel('Rho Value')
        ax3.set_ylabel('Mean Episode Reward')
        ax3.set_title('Rho vs Reward by Config')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Subplot 4: Statistics
        all_rewards = []
        for rho in self.rho_values:
            for config_name in config_names:
                if (config_name in checkpoint_data[checkpoint] and 
                    rho in checkpoint_data[checkpoint][config_name]):
                    all_rewards.append(checkpoint_data[checkpoint][config_name][rho])
        
        if all_rewards:
            stats_text = f"""
Statistics for Checkpoint {checkpoint}:
Total Configurations: {len(config_names)}
Total Rho Values: {len(self.rho_values)}
Mean Reward: {np.mean(all_rewards):.2f}
Std Reward: {np.std(all_rewards):.2f}
Min Reward: {np.min(all_rewards):.2f}
Max Reward: {np.max(all_rewards):.2f}
            """
            ax4.text(0.1, 0.5, stats_text, transform=ax4.transAxes, 
                    fontsize=10, verticalalignment='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
            ax4.set_xlim(0, 1)
            ax4.set_ylim(0, 1)
            ax4.axis('off')
            ax4.set_title('Summary Statistics')
        
        plt.tight_layout()
        
        # Save plot
        if save_dir is None:
            save_dir = self.results_dir
        
        save_path = Path(save_dir) / f'summary_ckpt{checkpoint}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved summary plot to {save_path}")
        
        plt.show()


def create_visualization_example():
    """
    Example usage of the RobustMARLVisualizer.
    """
    # Initialize visualizer
    visualizer = RobustMARLVisualizer(results_dir="results")
    
    # Set evaluation configs (modify these to match your actual configs)
    eval_configs = [
        {'name': 'Training_Config'},
        {'name': 'Hot_Humid_Miami'},
        {'name': 'Cold_Chicago'},
        {'name': 'Very_Hot_Humid_Honolulu'},
        {'name': 'Warm_Dry_ElPaso'},
        {'name': 'Cool_Marine_Seattle'},
    ]
    visualizer.set_eval_configs(eval_configs)
    
    # Load results for VDN algorithm
    try:
        visualizer.load_results('vdn', rho_values=[0.0, 0.0125])
        
        # Plot all checkpoints
        visualizer.plot_all_checkpoints()
        
        # Create summary plot for first checkpoint
        checkpoint_data = visualizer.extract_checkpoint_data()
        if checkpoint_data:
            first_checkpoint = min(checkpoint_data.keys())
            visualizer.create_summary_plot(first_checkpoint, checkpoint_data)
            
    except FileNotFoundError as e:
        print(f"Example failed: {e}")
        print("Make sure you have result files in the results directory")


if __name__ == "__main__":
    create_visualization_example() 