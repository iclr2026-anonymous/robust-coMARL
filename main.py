"""
Main training script for Robust MARL algorithms.
Supports VDN, QMIX, QTRAN, and their G-network variants.
"""

import argparse
import torch
from configs.config import get_rho_values, get_training_params
from algorithms.vdn import RobustVDN
from algorithms.qmix import RobustQMIX
from algorithms.qtran import RobustQTRAN
from algorithms.vdn_g import RobustVDNG
from algorithms.qmix_g import RobustQMIXG
from algorithms.qtran_g import RobustQTRANG
from algorithms.env_estimator_trainer import EnvEstimatorTrainer
from algorithms.vdn_groupdr import VDNGroupDR
from algorithms.qmix_groupdr import QMIXGroupDR
from algorithms.qtran_groupdr import QTRANGroupDR
from algorithms.vdn_dr import VDN_DOMAIN_RANDOMIZATION

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Robust MARL algorithms')
    parser.add_argument('--algorithm', type=str, default='vdn', 
                       choices=['vdn', 'qmix', 'qtran', 'vdn_g', 'qmix_g', 'qtran_g', 'env_estimator', 'vdn_groupdr', 'qmix_groupdr', 'qtran_groupdr','vdn_dr'],
                       help='Algorithm to train')
    parser.add_argument('--rho', type=float, default=None,
                       help='Specific rho value to train (if None, train all)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--behavior_checkpoint_dir', type=str, default=None,
                       help='For env_estimator: path to a VDN run dir containing network_parameter for fixed behavior policy')
    parser.add_argument('--env_estimator_path', type=str, default=None,
                       help='For groupdr: path to a trained env_estimator .pt file')
    
    args = parser.parse_args()
    
    # Set device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    print(f"Training algorithm: {args.algorithm}")
    
    # Get rho values to train (some algorithms ignore rho)
    if args.algorithm in ['env_estimator', 'vdn_groupdr']:
        rho_values = [args.rho] if args.rho is not None else [0.0]
    else:
        if args.rho is not None:
            rho_values = [args.rho]
        else:
            rho_values = get_rho_values(args.algorithm)
    
    print(f"Training with rho values: {rho_values}")
    
    # Train for each rho value
    for rho in rho_values:
        print(f"\n{'='*50}")
        print(f"Training {args.algorithm.upper()} with rho = {rho}")
        print(f"{'='*50}")
        
        try:
            if args.algorithm == 'vdn':
                agent = RobustVDN(rho_value=rho, device=device)
                results = agent.train()
            elif args.algorithm == 'qmix':
                agent = RobustQMIX(rho_value=rho, device=device)
                results = agent.train()
            elif args.algorithm == 'qtran':
                agent = RobustQTRAN(rho_value=rho, device=device)
                results = agent.train()
            elif args.algorithm == 'vdn_g':
                agent = RobustVDNG(rho_value=rho, device=device)
                results = agent.train()
            elif args.algorithm == 'qmix_g':
                agent = RobustQMIXG(rho_value=rho, device=device)
                results = agent.train()
            elif args.algorithm == 'qtran_g':
                agent = RobustQTRANG(rho_value=rho, device=device)
                results = agent.train()
            elif args.algorithm == 'env_estimator':
                if not args.behavior_checkpoint_dir:
                    raise ValueError('--behavior_checkpoint_dir is required for env_estimator')
                agent = EnvEstimatorTrainer(behavior_checkpoint_dir=args.behavior_checkpoint_dir, device=device)
                results = agent.train()
            elif args.algorithm == 'vdn_groupdr':
                if not args.env_estimator_path:
                    raise ValueError('--env_estimator_path is required for vdn_groupdr')
                agent = VDNGroupDR(env_estimator_path=args.env_estimator_path, device=device)
                results = agent.train()
            elif args.algorithm == 'qmix_groupdr':
                if not args.env_estimator_path:
                    raise ValueError('--env_estimator_path is required for qmix_groupdr')
                agent = QMIXGroupDR(env_estimator_path=args.env_estimator_path, device=device)
                results = agent.train()
            elif args.algorithm == 'qtran_groupdr':
                if not args.env_estimator_path:
                    raise ValueError('--env_estimator_path is required for qtran_groupdr')
                agent = QTRANGroupDR(env_estimator_path=args.env_estimator_path, device=device)
                results = agent.train()
            elif args.algorithm == 'vdn_dr':
                agent = VDN_DOMAIN_RANDOMIZATION(device=device)
                results = agent.train()
            else:
                print(f"Unknown algorithm: {args.algorithm}")
                continue
            
            print(f"Training completed for rho = {rho}")
            if isinstance(results, dict) and 'train_rewards' in results and results['train_rewards']:
                print(f"Final training reward: {results['train_rewards'][-1]:.4f}")
            elif isinstance(results, dict) and 'model_path' in results:
                print(f"Saved model to: {results['model_path']}")
            
        except Exception as e:
            print(f"Error training {args.algorithm} with rho = {rho}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\nTraining completed!")

if __name__ == '__main__':
    main() 