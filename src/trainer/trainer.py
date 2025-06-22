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
from collections import Counter
import time

# Import project modules
from src.models.LightGCN.model import LightGCN
from src.models.SimGCL.model import SimGCL
from src.models.XSimGCL.model import XSimGCL
from src.models.MixGCF.model import MixGCF
from src.models.TwinCL.model import TwinCL
from src.models.PopDRL.model import PopDRL
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
        self.run_id = (f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_"
                       f"{self.config['model_name']}_{self.config['dataset_name']}")
        self.current_run_log_dir = Path(config['log_dir']) / self.run_id
        self.current_run_log_dir.mkdir(parents=True, exist_ok=True)
        self.setup_logging(self.current_run_log_dir)
        self.setup_device(config['use_gpu'])

        # Load dataset
        self.dataset = RecDataset(
            config['dataset_path'],
            config['dataset_name'],
            n_candidate_negs=config.get('n_negs', 1),
            apply_ips_cn=config.get('ips_cn_enabled', False))

        # Create model
        self.model_name = config['model_name']
        if self.model_name == 'LightGCN':
            self.model = LightGCN(self.dataset, config)
        elif self.model_name == 'SimGCL':
            self.model = SimGCL(self.dataset, config)
        elif self.model_name == 'XSimGCL':
            self.model = XSimGCL(self.dataset, config)
        elif self.model_name == 'MixGCF':
            self.model = MixGCF(self.dataset, config)
        elif self.model_name == 'TwinCL':
            self.model = TwinCL(self.dataset, config)
        elif self.model_name == 'PopDRL':
            self.model = PopDRL(self.dataset, config)
        else:
            raise ValueError(f"Unknown model: {self.model_name}")

        self.model = self.model.to(self.device)

        # Setup optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=config['learning_rate']
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

        # Log initial model memory
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            initial_model_memory = torch.cuda.memory_allocated() / (1024 ** 3) # in GB
            self.logger.info(f"Initial model memory on GPU: {initial_model_memory:.2f} GB")
            self.tb_logger.log_scalar("memory/initial_model_gb", initial_model_memory, 0)

        # Check if item popularity was calculated (for metrics calculation)
        if self.dataset.item_popularity is None:
            self.dataset.item_popularity = self.dataset._compute_item_popularity()

        self.item_popularity = {i: pop for i, pop in enumerate(self.dataset.item_popularity)}

        self.item_groups = self._define_item_groups(
            head_percentage=config.get('head_percentage', 0.2)
        )
        self.logger.info(f"Items divided into groups: "
                         f"{len([i for i, g in self.item_groups.items() if g == 'head'])} in head, "
                         f"{len([i for i, g in self.item_groups.items() if g == 'tail'])} in tail.")
        
    def _define_item_groups(self, head_percentage: float) -> dict[int, str]:
        """
        Divides items into 'head' and 'tail' groups based on popularity.

        Args:
            head_percentage (float): The fraction of most popular items to be considered 'head'.

        Returns:
            A dictionary mapping item_id to its group ('head' or 'tail').
        """
        if not self.item_popularity:
            return {}

        # self.item_popularity это dict {item_id: pop_count}
        sorted_items = sorted(self.item_popularity.items(), key=lambda x: x[1], reverse=True)
        
        num_items = len(sorted_items)
        num_head_items = int(num_items * head_percentage)

        item_groups = {}
        for i, (item_id, _) in enumerate(sorted_items):
            if i < num_head_items:
                item_groups[item_id] = 'head'
            else:
                item_groups[item_id] = 'tail'
                
        return item_groups

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
        epoch_total_loss = 0
        total_samples = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch_idx, (users, pos_items, neg_items) in enumerate(pbar):
            users = users.to(self.device)
            pos_items = pos_items.to(self.device)
            neg_items = neg_items.to(self.device)

            batch_size = users.size(0)
            total_samples += batch_size

            loss_components = self.model.compute_loss(users, pos_items, neg_items)
            loss = loss_components["loss"]

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if isinstance(self.model, TwinCL):
                self.model.update_key_encoder()

            epoch_total_loss += loss.item() * batch_size

            formatted_log_postfix = {
                name: f"{val.item():.4f}" if isinstance(val, torch.Tensor) else f"{val:.4f}"
                for name, val in loss_components.items() if val is not None
            }
            pbar.set_postfix(formatted_log_postfix)

            # Log to tensorboard
            step = epoch * len(self.train_loader) + batch_idx
            for loss_type in ("loss", "bpr_loss", "reg_loss", "cl_loss", "proto_loss", "ssl_loss", "rec_loss"):
                if loss_type in loss_components and loss_components[loss_type] is not None:
                    self.tb_logger.log_scalar(f"train/{loss_type}", loss_components[loss_type].item(), step)

        avg_loss = epoch_total_loss / total_samples if total_samples > 0 else 0.0
        
        self.logger.info(f"Epoch {epoch} - Avg Loss: {avg_loss:.4f}")
        return avg_loss

    def _calculate_gini(self, item_counts: list[int]) -> float:
        """
        Calculate the Gini index for a list of item counts.
        Gini = (sum_i sum_j |x_i - x_j|) / (2 * n^2 * mean(x))
        """
        if not item_counts or sum(item_counts) == 0:
            return 0.0

        counts = np.array(item_counts, dtype=np.float32)
        n = len(counts)
        counts_sorted = np.sort(counts)

        index = np.arange(1, n + 1)
        numerator = np.sum((2 * index - n - 1) * counts_sorted)
        denominator = n * np.sum(counts_sorted)

        if denominator == 0:
            return 0.0

        return numerator / denominator

    def evaluate(self, epoch):
        """Evaluate model on test set"""
        self.model.eval()

        # Get all test users
        test_users = torch.LongTensor(list(self.dataset.test_data.keys())).to(self.device)

        # Evaluate in batches to avoid OOM
        metrics_results = {
            'precision': {k: 0 for k in self.topk}, 
            'recall': {k: 0 for k in self.topk}, 
            'ndcg': {k: 0 for k in self.topk},
            'hit_rate': {k: 0 for k in self.topk},
            'arp': {k: 0 for k in self.topk},
            'recall_head': {k: 0 for k in self.topk},
            'recall_tail': {k: 0 for k in self.topk},
            'ndcg_head': {k: 0 for k in self.topk},
            'ndcg_tail': {k: 0 for k in self.topk},
        }
        all_recommended_items_for_coverage = {k: set() for k in self.topk}
        total_rec_counts_for_gini = {k: Counter() for k in self.topk}

        num_users_with_head_items = {k: 0 for k in self.topk}
        num_users_with_tail_items = {k: 0 for k in self.topk}

        metrics_to_accumulate = [
                    'precision', 'recall', 'ndcg', 'hit_rate', 'arp',
                    'recall_head', 'recall_tail', 'ndcg_head', 'ndcg_tail'
        ]

        num_batches = (len(test_users) + self.eval_batch_size - 1) // self.eval_batch_size

        with torch.no_grad():
            for batch_idx in tqdm(range(num_batches), desc="Evaluating"):
                start_idx = batch_idx * self.eval_batch_size
                end_idx = min((batch_idx + 1) * self.eval_batch_size, len(test_users))
                batch_users = test_users[start_idx:end_idx]
                
                batch_metrics = self.model.calculate_metrics(
                    batch_users, 
                    self.dataset.test_data,
                    k_list=self.topk,
                    train_items=self.dataset.train_data,
                    all_item_ids=self.dataset.all_item_ids,
                    item_popularity=self.item_popularity,
                    item_groups=self.item_groups
                )

                # Accumulate metrics
                for metric_name in metrics_to_accumulate:
                    for k in self.topk:
                        metrics_results[metric_name][k] += batch_metrics[metric_name][k] * len(batch_users)

                for k in self.topk:
                    all_recommended_items_for_coverage[k].update(batch_metrics['coverage_items'][k])
                    total_rec_counts_for_gini[k].update(batch_metrics['rec_counts'][k])

        # Average metrics
        for metric_name in metrics_to_accumulate:
            for k in self.topk:
                metrics_results[metric_name][k] /= len(test_users)

        metrics_results['coverage'] = {k: 0 for k in self.topk}
        metrics_results['gini'] = {k: 0 for k in self.topk}

        total_items = len(self.dataset.all_item_ids)
        for k in self.topk:
            # Coverage
            metrics_results['coverage'][k] = len(all_recommended_items_for_coverage[k]) / total_items if total_items > 0 else 0
            # Gini Index
            item_counts = list(total_rec_counts_for_gini[k].values())
            metrics_results['gini'][k] = self._calculate_gini(item_counts)
                
        # Log metrics
        self.logger.info(f"Evaluation results for epoch {epoch}:")
        for k in self.topk:
            self.logger.info(f"--- Top-{k} ---")
            self.logger.info(f"Precision={metrics_results['precision'][k]:.4f}, "
                            f"Recall={metrics_results['recall'][k]:.4f}, "
                            f"NDCG={metrics_results['ndcg'][k]:.4f}, ")
            self.logger.info(f"Hit-Rate={metrics_results['hit_rate'][k]:.4f}, "
                            f"Coverage={metrics_results['coverage'][k]:.4f}, "
                            f"ARP={metrics_results['arp'][k]:.2f}, "
                            f"Gini={metrics_results['gini'][k]:.4f}")
            
            self.logger.info(f"Head Items: Recall={metrics_results['recall_head'][k]:.4f}, NDCG={metrics_results['ndcg_head'][k]:.4f}")
            self.logger.info(f"Tail Items: Recall={metrics_results['recall_tail'][k]:.4f}, NDCG={metrics_results['ndcg_tail'][k]:.4f}")
            
            # Log to tensorboard
            self.tb_logger.log_scalar(f'eval/precision@{k}', metrics_results['precision'][k], epoch)
            self.tb_logger.log_scalar(f'eval/recall@{k}', metrics_results['recall'][k], epoch)
            self.tb_logger.log_scalar(f'eval/ndcg@{k}', metrics_results['ndcg'][k], epoch)
            self.tb_logger.log_scalar(f'eval/hit_rate@{k}', metrics_results['hit_rate'][k], epoch)
            self.tb_logger.log_scalar(f'eval/coverage@{k}', metrics_results['coverage'][k], epoch)
            self.tb_logger.log_scalar(f'eval/arp@{k}', metrics_results['arp'][k], epoch) # Новая метрика
            self.tb_logger.log_scalar(f'eval/gini@{k}', metrics_results['gini'][k], epoch)

            self.tb_logger.log_scalar(f'eval/recall_head@{k}', metrics_results['recall_head'][k], epoch)
            self.tb_logger.log_scalar(f'eval/recall_tail@{k}', metrics_results['recall_tail'][k], epoch)
            self.tb_logger.log_scalar(f'eval/ndcg_head@{k}', metrics_results['ndcg_head'][k], epoch)
            self.tb_logger.log_scalar(f'eval/ndcg_tail@{k}', metrics_results['ndcg_tail'][k], epoch)


        return metrics_results
        
    def train(self):
        """Train model for multiple epochs"""
        best_recall = 0
        best_epoch = 0

        training_start_time = time.time()
        
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

        training_end_time = time.time()
        training_time = training_end_time - training_start_time

        self.logger.info(f"Training completed. Total training duration: {training_time:.2f}s")
        self.tb_logger.log_scalar("train/training_duration_seconds", training_time, self.config['epochs']) 
        self.logger.info(f"Best model at epoch {best_epoch} with Recall@{self.topk[0]}={best_recall:.4f}")
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
