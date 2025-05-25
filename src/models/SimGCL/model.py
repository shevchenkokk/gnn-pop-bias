import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Any, Optional, List
from src.utils.basic_dataset import BasicDataset
from src.models.BasicModel.model import BasicModel


class SimGCL(BasicModel):
    """
    SimGCL: Are Graph Augmentations Necessary? Simple Graph Contrastive Learning for Recommendation
    
    Reference: 
    Yu, J., Yin, H., Gao, M., Xia, X., Zhang, X., & Vucetic, S. (2022). 
    Are Graph Augmentations Necessary? Simple Graph Contrastive Learning for Recommendation. 
    SIGIR 2022.
    """
    def __init__(
        self,
        dataset: BasicDataset,
        config: dict,
    ):
        """
        Initialize SimGCL model

        Args:
            dataset: Dataset object containing user-item interactions
            config: Configuration dictionary with model parameters
        """
        super(SimGCL, self).__init__()
        self.dataset = dataset
        self.config = config
        self.n_users = self.dataset.n_users
        self.n_items = self.dataset.n_items
        self.embedding_dim = self.config["embedding_dim"]
        self.n_layers = self.config["SimGCL_n_layers"]
        self.decay = self.config.get("decay", 1e-4)
        self.device = self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        
        # SimGCL specific parameters
        self.cl_rate = self.config.get("cl_rate", 0.2)  # Weight for contrastive loss
        self.temp = self.config.get("temp", 0.2)  # Temperature for contrastive loss
        self.noise_scale = self.config.get("noise_scale", 0.1)  # Scale of noise for augmentation
        
        self.__init_weight()
        
    def __init_weight(self):
        """
        Initialize embeddings with normal distribution
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
            self.user_embs.weight.data.copy_(torch.from_numpy(self.config["user_embs"]))
            self.item_embs.weight.data.copy_(torch.from_numpy(self.config["item_embs"]))
            print("--- Use pretrained data ---")
        
        # Get the graph adjacency matrix in sparse format
        self.graph = self.dataset.get_sparse_graph().to(self.device)
        print("--- SimGCL is ready to go ---")
        
    def _add_noise(self, embeddings):
        """
        Add uniform noise to embeddings for augmentation
        
        Args:
            embeddings: Input embeddings
            
        Returns:
            Augmented embeddings with noise
        """
        noise = torch.rand_like(embeddings)
        noise_normalized = torch.sign(embeddings) * F.normalize(noise, dim=-1) * self.noise_scale
        return embeddings + noise_normalized

    def propagate(self, add_noise=False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Propagate methods for SimGCL
        
        Args:
            add_noise: Whether to add noise for augmentation
            
        Returns:
            Tuple of user and item embeddings after propagation
        """
        users_emb = self.user_embs.weight
        items_emb = self.item_embs.weight
        all_emb = torch.cat([users_emb, items_emb])
        
        layer_embeddings = [all_emb]
        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(self.graph, all_emb)
            if add_noise:
                all_emb = self._add_noise(all_emb)
            layer_embeddings.append(all_emb)
        
        # Mean aggregation of all layers
        layer_embeddings = torch.stack(layer_embeddings, dim=1)
        out = torch.mean(layer_embeddings, dim=1)
        
        users, items = torch.split(out, [self.n_users, self.n_items])
        return users, items
    
    def get_users_rating(self, users: torch.Tensor) -> torch.Tensor:
        """
        Compute item ratings for users
        
        Args:
            users: User indices tensor
            
        Returns:
            Rating matrix for specified users
        """
        all_users, all_items = self.propagate()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = torch.matmul(users_emb, items_emb.t())
        return rating
    
    def get_embedding(
        self,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get embeddings for users, positive items, and negative items
        
        Args:
            users: User indices tensor
            pos_items: Positive item indices tensor
            neg_items: Negative item indices tensor
            
        Returns:
            Tuple of embeddings for users, positive items, negative items,
            and their initial embeddings before propagation
        """
        all_users, all_items = self.propagate()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.user_embs(users)
        pos_emb_ego = self.item_embs(pos_items)
        neg_emb_ego = self.item_embs(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego
    
    def _create_contrastive_views(self):
        """
        Create two contrastive views with noise
        
        Returns:
            Two sets of user and item embeddings with different noise
        """
        users_emb1, items_emb1 = self.propagate(add_noise=True)
        users_emb2, items_emb2 = self.propagate(add_noise=True)
        
        return users_emb1, items_emb1, users_emb2, items_emb2
    
    def _contrastive_loss(self, view1, view2):
        """
        Compute contrastive loss between two views
        
        Args:
            view1: First view embeddings
            view2: Second view embeddings
            
        Returns:
            Contrastive loss
        """
        # Normalize embeddings
        view1 = nn.functional.normalize(view1, p=2, dim=1)
        view2 = nn.functional.normalize(view2, p=2, dim=1)
        
        # Compute similarity matrix
        pos_score = torch.sum(torch.mul(view1, view2), dim=1)
        pos_score = torch.exp(pos_score / self.temp)
        
        # Compute negative similarity
        ttl_score = torch.matmul(view1, view2.t())
        ttl_score = torch.exp(ttl_score / self.temp).sum(dim=1)
        
        # InfoNCE loss
        cl_loss = -torch.log(pos_score / ttl_score).mean()
        
        return cl_loss
    
    def compute_loss(self, users: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor) -> dict:
        """
        Compute BPR loss with contrastive learning
        
        Args:
            users: User indices tensor
            pos: Positive item indices tensor
            neg: Negative item indices tensor
            
        Returns:
            Dict of {total_loss, bpr_loss, reg_loss, cl_loss)
        """
        (users_emb, pos_emb, neg_emb, 
        userEmb0, posEmb0, negEmb0) = self.get_embedding(users.long(), pos.long(), neg.long())

        # BPR Loss
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        
        bpr_loss = - (pos_scores - neg_scores).sigmoid().log().mean()
        
        # Regularization Loss
        reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) +
                              posEmb0.norm(2).pow(2) +
                              negEmb0.norm(2).pow(2)) / float(len(users))
        
        # Contrastive Loss
        users_emb1, items_emb1, users_emb2, items_emb2 = self._create_contrastive_views()
        
        # Sample users and items for contrastive loss
        sampled_users = users.unique()
        sampled_items = torch.cat([pos, neg]).unique()
        
        u_cl_loss = self._contrastive_loss(
            users_emb1[sampled_users], 
            users_emb2[sampled_users]
        )
        
        i_cl_loss = self._contrastive_loss(
            items_emb1[sampled_items], 
            items_emb2[sampled_items]
        )
        
        cl_loss = u_cl_loss + i_cl_loss
        
        # Total Loss
        total_loss = bpr_loss + self.decay * reg_loss + self.cl_rate * cl_loss

        return {
            'total_loss': total_loss,
            'bpr_loss': bpr_loss,
            'reg_loss': reg_loss,
            'cl_loss': cl_loss
        }
    
    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for prediction
        
        Args:
            users: User indices tensor
            items: Item indices tensor
            
        Returns:
            Predicted scores (probabilities) for the user-item pairs
        """
        all_users, all_items = self.propagate()

        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_prod = torch.mul(users_emb, items_emb)
        return torch.sum(inner_prod, dim=1)