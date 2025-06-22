import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Any, Optional, List
from src.utils.basic_dataset import BasicDataset
from src.models.BasicModel.model import BasicModel


class MixGCF(BasicModel):
    """
    MixGCF: An Improved Training Method for Graph Neural Network-based Recommender Systems
    
    Reference: 
    Huang, T., Dong, Y., Cheng, M., Peng, Z., & Zhang, Y. (2021). 
    MixGCF: An Improved Training Method for Graph Neural Network-based Recommender Systems. 
    KDD 2021.
    """
    def __init__(
        self,
        dataset: BasicDataset,
        config: dict,
    ):
        """
        Initialize MixGCF model
        
        Args:
            dataset: Dataset object containing user-item interactions
            config: Configuration dictionary with model parameters
        """
        super(MixGCF, self).__init__()
        self.dataset = dataset
        self.config = config
        self.n_users = self.dataset.n_users
        self.n_items = self.dataset.n_items
        self.embedding_dim = self.config["embedding_dim"]
        self.n_layers = self.config["MixGCF_n_layers"]
        self.decay = self.config.get("decay", 1e-4)
        self.device = self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        
        # MixGCF specific parameters
        # n_candidate_negs (M in the paper): Number of candidate negative samples for synthesis per positive pair.
        self.n_candidate_negs = self.config.get("n_negs", 10) 
        
        self.__init_weight()
        
    def __init_weight(self):
        """
        Initialize embeddings with Xavier Uniform distribution.
        """
        self.user_embs = nn.Embedding(
            num_embeddings=self.n_users, embedding_dim=self.embedding_dim)
        self.item_embs = nn.Embedding(
            num_embeddings=self.n_items, embedding_dim=self.embedding_dim)

        if "pretrain" not in self.config or not self.config["pretrain"]:
            nn.init.xavier_uniform_(self.user_embs.weight)
            nn.init.xavier_uniform_(self.item_embs.weight)
            print("--- Use Xavier Uniform initializer ---")
        else:
            # Load pretrained embeddings if specified in config
            self.user_embs.weight.data.copy_(torch.from_numpy(self.config["user_embs"]))
            self.item_embs.weight.data.copy_(torch.from_numpy(self.config["item_embs"]))
            print("--- Use pretrained data ---")

        self.graph = self.dataset.get_sparse_graph().to(self.device)
        print("--- MixGCF is ready to go ---")
        
    def propagate(self) -> Tuple:
        """
        Perform standard LightGCN propagation for all users and items.
        Returns final aggregated embeddings and a list of layer-wise embeddings.
        
        Returns:
            Tuple:
                - final_users: Final aggregated user embeddings.
                - final_items: Final aggregated item embeddings.
                - all_layer_embeddings: List of tensors, where each tensor contains
                                        all user and item embeddings for a specific layer (0 to L).
        """
        # Get initial (ego) embeddings
        users_emb = self.user_embs.weight
        items_emb = self.item_embs.weight
        all_emb = torch.cat([users_emb, items_emb])
        
        # Perform layer-wise propagation and store embeddings from each layer
        all_layer_embeddings = [all_emb] # Layer 0 (ego embeddings)
        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(self.graph, all_emb)
            all_layer_embeddings.append(all_emb) # Layer 1, 2,..., L
        
        # Mean aggregation of all layers for final embeddings (LightGCN output)
        final_all_embeddings = torch.mean(torch.stack(all_layer_embeddings, dim=1), dim=1)
        
        final_users, final_items = torch.split(final_all_embeddings, [self.n_users, self.n_items])
        
        return final_users, final_items, all_layer_embeddings
    
    def _synthesize_negative_embedding_batched(
        self,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        candidate_neg_ids_batch: torch.Tensor,
        all_layer_embeddings: List
    ) -> torch.Tensor:
        """
        Synthesizes a hard negative embedding for a given (user, positive_item) pair
        using Positive Mixing and Hop Mixing, as described in MixGCF.
        
        Args:
            user_id: A batch of user indices.
            pos_items: A batch of positive item indices.
            candidate_neg_ids_batch: A batch of candidate negative item indices.
            all_layer_embeddings: A list of tensors, where each tensor contains
                                  all user and item embeddings for a specific layer.
                                  (e.g., [layer0_embs, layer1_embs,..., layerL_embs])
        Returns:
            A tensor of synthetic hard negative embeddings
        """
        num_layers = len(all_layer_embeddings) # L+1
        batch_size = users.shape[0]
        
        # 1. Pre-fetch all necessary embeddings for the entire batch and all layers
        # user_embs_all_layers: List of (batch_size, embedding_dim) tensors, one for each layer
        user_embs_all_layers = [layer_emb[users] for layer_emb in all_layer_embeddings]
        # pos_item_embs_all_layers: List of (batch_size, embedding_dim) tensors
        pos_item_embs_all_layers = [layer_emb[self.n_users + pos_items] for layer_emb in all_layer_embeddings]
        
        # candidate_neg_embs_all_layers: List of (batch_size, n_candidate_negs, embedding_dim) tensors
        # For each layer, get embeddings for all candidates for all batch samples
        # We need to access items using `self.n_users + candidate_neg_ids_batch.view(-1)`
        # then reshape to (batch_size, n_candidate_negs, embedding_dim)
        candidate_neg_embs_all_layers = []
        for layer_emb in all_layer_embeddings:
            # Get embeddings for all candidates from all batch items
            # shape will be (batch_size * n_candidate_negs, embedding_dim)
            flat_candidate_embs = layer_emb[self.n_users + candidate_neg_ids_batch.view(-1)]
            # Reshape to (batch_size, n_candidate_negs, embedding_dim)
            candidate_neg_embs_all_layers.append(flat_candidate_embs.view(
                batch_size, self.n_candidate_negs, self.embedding_dim
            ))

        selected_neg_embs_all_layers = [] # To store the selected hard negative from each layer, shape (batch_size, embedding_dim)
        
        for l in range(num_layers):
            # alpha_l: (1,) or (batch_size, 1) if independent alphas per batch item
            # For MixGCF, it's typically one alpha per layer
            alpha_l = torch.rand(1, device=self.device) 
            
            # Positive Mixing (Vectorized)
            # mixed_embs_at_layer_l: (batch_size, n_candidate_negs, embedding_dim)
            # (1 - alpha_l) broadcasted across (batch_size, n_candidate_negs, embedding_dim)
            mixed_embs_at_layer_l = alpha_l * pos_item_embs_all_layers[l].unsqueeze(1) + \
                                    (1 - alpha_l) * candidate_neg_embs_all_layers[l]
            
            # Hop Mixing (Vectorized)
            # scores: (batch_size, n_candidate_negs)
            scores = torch.sum(user_embs_all_layers[l].unsqueeze(1) * mixed_embs_at_layer_l, dim=2)
            
            # hardest_idx: (batch_size,) - indices of the hardest negative for each sample in batch
            hardest_idx = torch.argmax(scores, dim=1)
            
            # Use gather to select the hardest mixed negative embedding for each batch item
            # Expand hardest_idx to match the dimensions of mixed_embs_at_layer_l
            hardest_idx_expanded = hardest_idx.unsqueeze(1).unsqueeze(2).expand(-1, -1, self.embedding_dim)
            
            # selected_embs: (batch_size, 1, embedding_dim) -> squeeze to (batch_size, embedding_dim)
            selected_embs = torch.gather(mixed_embs_at_layer_l, 1, hardest_idx_expanded).squeeze(1)
            
            selected_neg_embs_all_layers.append(selected_embs)

        # Final Pooling (Vectorized)
        # Stack: (num_layers, batch_size, embedding_dim)
        # Mean: (batch_size, embedding_dim)
        synthetic_neg_embs = torch.mean(torch.stack(selected_neg_embs_all_layers, dim=0), dim=0)
        
        return synthetic_neg_embs

    def get_users_rating(self, users: torch.Tensor) -> torch.Tensor:
        """
        Compute item ratings for specified users using the final propagated embeddings.
        
        Args:
            users: User indices tensor.
            
        Returns:
            Rating matrix for specified users.
        """
        all_users, all_items, _ = self.propagate() 
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = torch.matmul(users_emb, items_emb.t())
        return rating
    
    def get_embedding(
        self,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor
    ) -> Tuple:
        """
        Get final propagated embeddings for users, positive items, and raw negative items,
        along with their initial ego embeddings. This method is primarily for evaluation/logging
        and returns propagated embeddings of *raw* negatives, not synthesized ones.
        
        Args:
            users: User indices tensor.
            pos_items: Positive item indices tensor.
            neg_items: Raw negative item indices tensor.
            
        Returns:
            Tuple of:
                - users_emb: Propagated user embeddings.
                - pos_emb: Propagated positive item embeddings.
                - neg_emb: Propagated raw negative item embeddings.
                - users_emb_ego: Initial (ego) user embeddings.
                - pos_emb_ego: Initial (ego) positive item embeddings.
                - neg_emb_ego: Initial (ego) raw negative item embeddings.
        """
        all_users, all_items, _ = self.propagate() 
        
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        
        users_emb_ego = self.user_embs(users)
        pos_emb_ego = self.item_embs(pos_items)
        neg_emb_ego = self.item_embs(neg_items)
        
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego
    
    def compute_loss(self, users: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor) -> dict:
        """
        Compute BPR loss with MixGCF's synthesized hard negatives and regularization loss.
        
        Args:
            users: User indices tensor (batch_size).
            pos: Positive item indices tensor (batch_size).
            neg: Raw negative item indices tensor (batch_size * n_candidate_negs).
                 This 'neg' tensor should contain the M candidate negatives for each positive.
                 For example, if batch_size=B, then neg should have B * self.n_candidate_negs elements,
                 where the first self.n_candidate_negs elements are candidates for the first (user, pos) pair,
                 the next self.n_candidate_negs elements for the second pair, and so on.
            
        Returns:
            Dictionary containing total_loss, bpr_loss, reg_loss.
        """
        # Perform standard LightGCN propagation to get all final and layer-wise embeddings
        final_users_emb, final_items_emb, all_layer_embeddings = self.propagate()
        
        # Extract final propagated embeddings for the current batch
        users_emb = final_users_emb[users]
        pos_emb = final_items_emb[pos]
        
        # Get initial (ego) embeddings for regularization
        userEmb0 = self.user_embs(users)
        posEmb0 = self.item_embs(pos)
        # For regularization loss, we use the ego embeddings of the *raw* negative samples.
        # The 'neg' tensor contains all raw candidates sampled for the batch.
        negEmb0 = self.item_embs(neg) 

        # Synthesize hard negative embeddings using MixGCF logic (fully vectorized)
        batch_size = users.shape[0]
        # Reshape `neg` tensor for batched processing in _synthesize_negative_embedding_batched
        candidate_neg_ids_batch = neg.view(batch_size, self.n_candidate_negs)
            
        synthetic_neg_emb = self._synthesize_negative_embedding_batched(
            users, pos, candidate_neg_ids_batch, all_layer_embeddings
        )
        
        # BPR Loss with synthesized negatives
        pos_scores = torch.sum(torch.mul(users_emb, pos_emb), dim=1)
        neg_scores = torch.sum(torch.mul(users_emb, synthetic_neg_emb), dim=1) # Use synthetic negatives

        bpr_loss = - (pos_scores - neg_scores).sigmoid().log().mean()

        # Regularization Loss (applied to ego embeddings)
        # This averages the regularization loss over the batch size.
        reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) +
                              posEmb0.norm(2).pow(2) +
                              negEmb0.norm(2).pow(2)) / float(batch_size)

        # Total Loss
        total_loss = bpr_loss + self.decay * reg_loss

        return {
            "loss": total_loss,
            "bpr_loss": bpr_loss,
            "reg_loss": reg_loss,
        }
    
    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for prediction (inference).
        Uses the final aggregated embeddings from standard LightGCN propagation.
        
        Args:
            users: User indices tensor.
            items: Item indices tensor.
            
        Returns:
            Predicted scores (inner product) for the user-item pairs.
        """
        all_users, all_items, _ = self.propagate() 

        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_prod = torch.mul(users_emb, items_emb)
        return torch.sum(inner_prod, dim=1)