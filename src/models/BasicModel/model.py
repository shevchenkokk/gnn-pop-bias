import torch
import numpy as np
from typing import Dict, List, Tuple, Any

class BasicModel(torch.nn.Module):    
    def __init__(self):
        super(BasicModel, self).__init__()
    
    def get_users_rating(self, users):
        """
        Get ratings for users
        
        Args:
            users: User IDs
            
        Returns:
            Ratings for all items for the given users
        """
        raise NotImplementedError
    
    def calculate_metrics(self, 
                         user_ids: torch.Tensor, 
                         test_items: Dict[int, List[int]], 
                         k_list: List[int],
                         train_items: Dict[int, List[int]] = None) -> Dict[str, Dict[int, float]]:
        """
        Calculate evaluation metrics for the model
        
        Args:
            user_ids: Tensor of user IDs to evaluate
            test_items: Dictionary mapping user IDs to lists of test item IDs
            k_list: List of k values for top-k metrics
            train_items: Dictionary mapping user IDs to lists of train item IDs (optional)
            
        Returns:
            Dictionary of metrics with format:
            {
                'precision': {k1: value1, k2: value2, ...},
                'recall': {k1: value1, k2: value2, ...},
                'ndcg': {k1: value1, k2: value2, ...}
            }
        """
        # Initialize metrics
        precision = {k: [] for k in k_list}
        recall = {k: [] for k in k_list}
        ndcg = {k: [] for k in k_list}
        
        # Get ratings for all users
        ratings = self.get_users_rating(user_ids)
        
        # For each user
        for i, user in enumerate(user_ids):
            user_id = user.item()
            
            # Get test and train items for this user
            test_item_list = test_items.get(user_id, [])
            
            if len(test_item_list) == 0:
                continue
                
            # Set ratings of train items to -inf to exclude them from recommendations
            user_ratings = ratings[i].clone()
            if train_items is not None:
                train_item_list = train_items.get(user_id, [])
                if len(train_item_list) > 0:
                    train_items_tensor = torch.LongTensor(train_item_list).to(user_ratings.device)
                    user_ratings[train_items_tensor] = -float('inf')
            
            # Get top-k item indices
            _, top_indices = torch.topk(user_ratings, max(k_list))
            top_indices = top_indices.cpu().numpy()
            
            # Convert test items to set for faster lookup
            test_items_set = set(test_item_list)
            
            # Calculate metrics for each k
            for k in k_list:
                top_k_indices = top_indices[:k]
                
                # Count number of test items in top-k recommendations
                num_hits = len(set(top_k_indices) & test_items_set)
                
                # Precision@k = (# of recommended items that are relevant) / (# of recommended items)
                precision[k].append(num_hits / k)
                
                # Recall@k = (# of recommended items that are relevant) / (# of relevant items)
                recall[k].append(num_hits / len(test_items_set))
                
                # NDCG@k
                dcg = 0
                idcg = 0
                
                # Calculate DCG
                for idx, item in enumerate(top_k_indices):
                    if item in test_items_set:
                        # Using binary relevance (1 if relevant, 0 if not)
                        # DCG formula: sum(rel_i / log2(i+2))
                        dcg += 1 / np.log2(idx + 2)
                
                # Calculate IDCG (ideal DCG)
                # IDCG is the DCG for the best possible ranking (all relevant items at the top)
                for idx in range(min(len(test_items_set), k)):
                    idcg += 1 / np.log2(idx + 2)
                
                # NDCG = DCG / IDCG
                if idcg > 0:
                    ndcg[k].append(dcg / idcg)
                else:
                    ndcg[k].append(0)
        
        # Calculate average metrics
        for k in k_list:
            precision[k] = np.mean(precision[k]) if precision[k] else 0
            recall[k] = np.mean(recall[k]) if recall[k] else 0
            ndcg[k] = np.mean(ndcg[k]) if ndcg[k] else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'ndcg': ndcg
        }
