import torch
import numpy as np
from typing import Dict, List, Tuple, Any
from collections import Counter

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
                         train_items: Dict[int, List[int]] = None,
                         all_item_ids: List[int] = None,
                         item_popularity: Dict[int, int] = None,
                         item_groups: Dict[int, str] = None) -> Dict[str, Dict[int, Any]]:
        """
        Calculate evaluation metrics for the model
        
        Args:
            user_ids: Tensor of user IDs to evaluate
            test_items: Dictionary mapping user IDs to lists of test item IDs
            k_list: List of k values for top-k metrics
            train_items: Dictionary mapping user IDs to lists of train item IDs (optional)
            all_item_ids: List of all unique item IDs in the dataset (optional, needed for Coverage)
            item_popularity: Dictionary mapping item IDs to their popularity count.
            item_groups: Dictionary mapping item IDs to their group.
            
        Returns:
            Dictionary of metrics with format:
            {
                'precision': {k1: value1, k2: value2, ...},
                'recall': {k1: value1, k2: value2, ...},
                'ndcg': {k1: value1, k2: value2, ...},
                'hit_rate': {k1: value1, k2: value2, ...},
                'coverage': {k1: value1, k2: value2, ...},
                'arp': {k1: value1, k2: value2, ...},
                'gini': {k1: value1, k2: value2, ...},
            }
        """
        # Initialize metrics
        precision = {k: [] for k in k_list}
        recall = {k: [] for k in k_list}
        ndcg = {k: [] for k in k_list}
        hit_rate = {k: [] for k in k_list}
        all_recommended_items = {k: set() for k in k_list}

        arp = {k: [] for k in k_list} # For ARP
        batch_rec_counts = {k: Counter() for k in k_list}

        # For groups
        recall_head = {k: [] for k in k_list}
        recall_tail = {k: [] for k in k_list}
        ndcg_head = {k: [] for k in k_list}
        ndcg_tail = {k: [] for k in k_list}

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
            if not test_items_set:
                continue

            test_items_head = {item for item in test_items_set if item_groups.get(item) == 'head'}
            test_items_tail = {item for item in test_items_set if item_groups.get(item) == 'tail'}
            
            # Calculate metrics for each k
            for k in k_list:
                top_k_indices = top_indices[:k]
                top_k_set = set(top_k_indices)
                
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

                # Hit-Rate@k
                if num_hits > 0:
                    hit_rate[k].append(1)
                else:
                    hit_rate[k].append(0)

                all_recommended_items[k].update(top_k_indices)
                if item_popularity:
                    k_pop = [item_popularity.get(rec_item, 0) for rec_item in top_k_indices]
                    arp[k].append(np.mean(k_pop) if k_pop else 0)

                batch_rec_counts[k].update(top_k_indices)

                # Recall for groups
                if test_items_head:
                    hits_head = len(top_k_set & test_items_head)
                    recall_head[k].append(hits_head / len(test_items_head))

                if test_items_tail:
                    hits_tail = len(top_k_set & test_items_tail)
                    recall_tail[k].append(hits_tail / len(test_items_tail))

                # NDCG for groups
                if test_items_head:
                    dcg_head, idcg_head = 0, 0
                    for idx, item in enumerate(top_k_indices):
                        if item in test_items_head:
                            dcg_head += 1 / np.log2(idx + 2)
                    for idx in range(min(len(test_items_head), k)):
                        idcg_head += 1 / np.log2(idx + 2)
                    ndcg_head[k].append(dcg_head / idcg_head if idcg_head > 0 else 0)

                if test_items_tail:
                    dcg_tail, idcg_tail = 0, 0
                    for idx, item in enumerate(top_k_indices):
                        if item in test_items_tail:
                            dcg_tail += 1 / np.log2(idx + 2)
                    for idx in range(min(len(test_items_tail), k)):
                        idcg_tail += 1 / np.log2(idx + 2)
                    ndcg_tail[k].append(dcg_tail / idcg_tail if idcg_tail > 0 else 0)
                

        # Calculate average metrics
        for k in k_list:
            precision[k] = np.mean(precision[k]) if precision[k] else 0
            recall[k] = np.mean(recall[k]) if recall[k] else 0
            ndcg[k] = np.mean(ndcg[k]) if ndcg[k] else 0
            hit_rate[k] = np.mean(hit_rate[k]) if hit_rate[k] else 0
            arp[k] = np.mean(arp[k]) if arp[k] else 0
            recall_head[k] = np.mean(recall_head[k]) if recall_head[k] else 0
            recall_tail[k] = np.mean(recall_tail[k]) if recall_tail[k] else 0
            ndcg_head[k] = np.mean(ndcg_head[k]) if ndcg_head[k] else 0
            ndcg_tail[k] = np.mean(ndcg_tail[k]) if ndcg_tail[k] else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'ndcg': ndcg,
            'hit_rate': hit_rate,
            'coverage_items': all_recommended_items,
            'arp': arp,
            'rec_counts': batch_rec_counts,
            'recall_head': recall_head,
            'recall_tail': recall_tail,
            'ndcg_head': ndcg_head,
            'ndcg_tail': ndcg_tail,
        }