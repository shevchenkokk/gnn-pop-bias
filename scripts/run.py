import argparse
import os
from pathlib import Path
import yaml
import torch
from src.trainer.trainer import Trainer


def main():
    """Main function to run training or evaluation"""
    parser = argparse.ArgumentParser(description="Run training or evaluation")
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--mode', type=str, choices=['train', 'eval'], default='train', 
                        help='Run mode: train or eval')
    parser.add_argument('--checkpoint', type=str, default=None, 
                        help='Path to checkpoint file for evaluation')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create trainer
    trainer = Trainer(config)
    
    if args.mode == 'train':
        # Train model
        trainer.train()
    else:
        # Evaluate model
        if args.checkpoint is None:
            raise ValueError("Checkpoint path must be provided for evaluation mode")
        
        # Load checkpoint
        trainer.load_checkpoint(args.checkpoint)
        
        # Evaluate
        metrics = trainer.evaluate(0)  # 0 is a placeholder for epoch
        
        print("Evaluation results:")
        for k in config['topk']:
            print(f"Top-{k}: Precision={metrics['precision'][k]:.4f}, "
                  f"Recall={metrics['recall'][k]:.4f}, "
                  f"NDCG={metrics['ndcg'][k]:.4f}")


if __name__ == '__main__':
    main()
