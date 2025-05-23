import os
import torch
import numpy as np
import random
from pathlib import Path
from datetime import datetime
import logging
import yaml
import argparse
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm

# Import project modules
from src.models.LightGCN.model import LightGCN
from src.utils.data_loader import RecDataset
from src.utils.logger import setup_logger, TensorboardLogger

class Trainer:
    """
    Trainer class for recommendation models
    """
    def __init__(self, config):
        """
        Initialize trainer with configuration
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.setup_seed(config['seed'])
        self.run_id = datetime.now().strftime('%Y%m%d_%H%M%S') + f"_{config['model_name']}"
        self.current_run_log_dir = Path(config['log_dir']) / self.run_id
        self.current_run_log_dir.mkdir(parents=True, exist_ok=True)
        self.setup_logging(self.current_run_log_dir)
        self.setup_device(config['use_gpu'])
        
        # Load dataset
        self.dataset = RecDataset(config['dataset_path'], config['dataset_name'])
        
        # Create model
        self.model_name = config['model_name']
        if self.model_name == 'LightGCN':
            self.model = LightGCN(self.dataset, config)
        elif self.model_name == 'LightGCN_IPS':
            self.model = LightGCN_IPS(self.dataset, config)
        elif self.model_name == 'MixGCF':
            from src.models.MixGCF.model import MixGCF
            self.model = MixGCF(self.dataset, config)
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
        
        self.model = self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=config['learning_rate'],
            weight_decay=0
        )
        
        # Setup dataloader
        self.train_loader = DataLoader(
            self.dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config['num_workers']
        )
        
        # Setup evaluation
        self.eval_batch_size = config['eval_batch_size']
        self.topk = config['topk']
        
        # Setup checkpointing
        self.checkpoint_dir = Path(config['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup tensorboard
        self.tb_logger = TensorboardLogger(self.current_run_log_dir)
        
    def setup_seed(self, seed):
        """Set random seed for reproducibility"""
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        
    def setup_device(self, use_gpu):
        """Setup computation device"""
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
        self.logger.info(f"Using device: {self.device}")
        
    def setup_logging(self, log_dir):
        """Setup logging"""
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logger(
            self.config['model_name'], 
            log_dir / f"{self.config['model_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        
    def save_checkpoint(self, epoch, metrics=None):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'metrics': metrics
        }
        
        checkpoint_path = self.checkpoint_dir / f"{self.model_name}_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved to {checkpoint_path}")
        
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.logger.info(f"Loaded checkpoint from {checkpoint_path} (epoch {checkpoint['epoch']})")
        return checkpoint['epoch']
        
    def train_epoch(self, epoch):
        """Train model for one epoch"""
        self.model.train()
        total_loss = 0
        total_reg_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch_idx, (users, pos_items, neg_items) in enumerate(pbar):
            users = users.to(self.device)
            pos_items = pos_items.to(self.device)
            neg_items = neg_items.to(self.device)
            
            self.optimizer.zero_grad()
            loss, reg_loss = self.model.bpr_loss(users, pos_items, neg_items)
            total_loss = loss + reg_loss
            total_loss.backward()
            self.optimizer.step()
            
            pbar.set_postfix({
                'bpr_loss': loss.item(),
                'reg_loss': reg_loss.item(),
                'total_loss': total_loss.item()
            })
            
            total_loss += loss.item()
            total_reg_loss += reg_loss.item()
            
            # Log to tensorboard
            step = epoch * len(self.train_loader) + batch_idx
            self.tb_logger.log_scalar('train/bpr_loss', loss.item(), step)
            self.tb_logger.log_scalar('train/reg_loss', reg_loss.item(), step)
            self.tb_logger.log_scalar('train/total_loss', total_loss.item(), step)
            
        avg_loss = total_loss / len(self.train_loader)
        avg_reg_loss = total_reg_loss / len(self.train_loader)
        
        self.logger.info(f"Epoch {epoch} - Avg Loss: {avg_loss:.4f}, Avg Reg Loss: {avg_reg_loss:.4f}")
        return avg_loss
        
    def evaluate(self, epoch):
        """Evaluate model on test set"""
        self.model.eval()
        
        # Get all test users
        test_users = torch.LongTensor(list(self.dataset.test_data.keys())).to(self.device)
        
        # Evaluate in batches to avoid OOM
        metrics_results = {'precision': {}, 'recall': {}, 'ndcg': {}}
        for k in self.topk:
            metrics_results['precision'][k] = 0
            metrics_results['recall'][k] = 0
            metrics_results['ndcg'][k] = 0
            
        num_batches = (len(test_users) + self.eval_batch_size - 1) // self.eval_batch_size
        
        with torch.no_grad():
            for batch_idx in tqdm(range(num_batches), desc="Evaluating"):
                start_idx = batch_idx * self.eval_batch_size
                end_idx = min((batch_idx + 1) * self.eval_batch_size, len(test_users))
                batch_users = test_users[start_idx:end_idx]
                
                batch_metrics = self.model.calculate_metrics(
                    batch_users, 
                    self.dataset.test_data,
                    k_list=self.topk
                )
                
                # Accumulate metrics
                for metric in ['precision', 'recall', 'ndcg']:
                    for k in self.topk:
                        metrics_results[metric][k] += batch_metrics[metric][k] * len(batch_users)
        
        # Average metrics
        for metric in ['precision', 'recall', 'ndcg']:
            for k in self.topk:
                metrics_results[metric][k] /= len(test_users)
                
        # Log metrics
        self.logger.info(f"Evaluation results for epoch {epoch}:")
        for k in self.topk:
            self.logger.info(f"Top-{k}: Precision={metrics_results['precision'][k]:.4f}, "
                            f"Recall={metrics_results['recall'][k]:.4f}, "
                            f"NDCG={metrics_results['ndcg'][k]:.4f}")
            
            # Log to tensorboard
            self.tb_logger.log_scalar(f'eval/precision@{k}', metrics_results['precision'][k], epoch)
            self.tb_logger.log_scalar(f'eval/recall@{k}', metrics_results['recall'][k], epoch)
            self.tb_logger.log_scalar(f'eval/ndcg@{k}', metrics_results['ndcg'][k], epoch)
            
        return metrics_results
        
    def train(self):
        """Train model for multiple epochs"""
        best_recall = 0
        best_epoch = 0
        
        for epoch in range(1, self.config['epochs'] + 1):
            # Train for one epoch
            self.train_epoch(epoch)
            
            # Evaluate
            if epoch % self.config['eval_freq'] == 0:
                metrics = self.evaluate(epoch)
                
                # Save checkpoint
                self.save_checkpoint(epoch, metrics)
                
                # Check if best model
                recall = metrics['recall'][self.topk[0]]
                if recall > best_recall:
                    best_recall = recall
                    best_epoch = epoch
                    # Save best model
                    checkpoint_path = self.checkpoint_dir / f"{self.model_name}_best.pt"
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'config': self.config,
                        'metrics': metrics
                    }, checkpoint_path)
                    self.logger.info(f"New best model saved at epoch {epoch}")
                    
        self.logger.info(f"Training completed. Best model at epoch {best_epoch} with Recall@{self.topk[0]}={best_recall:.4f}")
        return best_epoch, best_recall

def main():
    """Main function to run training"""
    parser = argparse.ArgumentParser(description="Train recommendation model")
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create trainer and train
    trainer = Trainer(config)
    trainer.train()

if __name__ == '__main__':
    main()
