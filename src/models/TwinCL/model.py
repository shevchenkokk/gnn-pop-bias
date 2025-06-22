import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Any, Optional, List
from src.utils.basic_dataset import BasicDataset
from src.models.BasicModel.model import BasicModel


class TwinCL(BasicModel):
    """
    TwinCL: A Twin Graph Contrastive Learning Model for Collaborative Filtering

    Reference:
    Chengkai Liu, Jianling Wang, James Caverlee (2024).
    TwinCL: A Twin Graph Contrastive Learning Model for Collaborative Filtering.
    """
    def __init__(
        self,
        dataset: BasicDataset,
        config: dict,
    ):
        """
        Initializes the TwinCL model.

        Args:
            dataset: The dataset object containing user-item interactions.
            config: A dictionary containing model configurations and hyperparameters.
        """
        super(TwinCL, self).__init__()
        self.dataset = dataset
        self.config = config
        self.n_users = self.dataset.n_users
        self.n_items = self.dataset.n_items
        self.embedding_dim = self.config["embedding_dim"]
        self.n_layers = self.config["TwinCL_n_layers"]
        self.decay = self.config.get("decay", 1e-4)
        self.device = self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        
        # TwinCL specific parameters
        self.mcl_rate = self.config.get("mcl_rate", 0.2) # Weight for momentum contrastive loss
        self.temp = self.config.get("temp", 0.2) # Temperature for contrastive loss
        self.gamma = self.config.get("gamma", 0.1) # Weight for uniformity loss
        self.m = self.config.get("m", 0.999) # Momentum coefficient for key encoder update

        self.__init_weight()
        
        # Initialize key model
        self.key_model = TwinCL_Encoder(self.n_users, self.n_items, self.embedding_dim, self.n_layers)
        self.key_model.set_graph(self.graph)
        self.key_model.load_state_dict(self.model.state_dict())
        self.key_model.to(self.device)
        self.key_model.eval()

    def __init_weight(self):
        """
        Initializes the model's weights and sets up the graph.
        """
        self.model = TwinCL_Encoder(self.n_users, self.n_items, self.embedding_dim, self.n_layers)
        self.graph = self.dataset.get_sparse_graph().to(self.device)
        self.model.set_graph(self.graph)
        self.model.to(self.device)
        print("--- TwinCL is ready to go ---")

    def propagate(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Propagates embeddings through the graph neural network.

        Returns:
            A tuple containing user embeddings and item embeddings after propagation.
        """
        return self.model()

    def get_users_rating(self, users: torch.Tensor) -> torch.Tensor:
        """
        Calculates the predicted ratings for specified users.

        Args:
            users: A tensor of user indices.

        Returns:
            A tensor representing the predicted ratings for the given users across all items.
        """
        all_users, all_items = self.propagate()
        users_emb = all_users[users.long()]
        items_emb = all_items
        return torch.matmul(users_emb, items_emb.t())

    def get_embedding(self, users, pos_items, neg_items):
        """
        Retrieves embeddings for users, positive items, and negative items.

        Args:
            users: A tensor of user indices.
            pos_items: A tensor of positive item indices.
            neg_items: A tensor of negative item indices.

        Returns:
            A tuple containing:
            - users_emb: Propagated embeddings for users.
            - pos_emb: Propagated embeddings for positive items.
            - neg_emb: Propagated embeddings for negative items.
            - users_emb_ego: Initial (ego) embeddings for users.
            - pos_emb_ego: Initial (ego) embeddings for positive items.
            - neg_emb_ego: Initial (ego) embeddings for negative items.
        """
        all_users, all_items = self.propagate()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.model.get_ego_embeddings()[0][users]
        pos_emb_ego = self.model.get_ego_embeddings()[1][pos_items]
        neg_emb_ego = self.model.get_ego_embeddings()[1][neg_items]
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def cal_momentum_loss(self, users: torch.Tensor, pos_items: torch.Tensor) -> torch.Tensor:
        """
        Calculates the momentum contrastive loss.

        Args:
            users: A tensor of user indices.
            pos_items: A tensor of positive item indices.

        Returns:
            The combined user and item momentum contrastive loss.
        """
        u_idx = torch.unique(users)
        i_idx = torch.unique(pos_items)
        
        user_view_1, item_view_1 = self.model()
        with torch.no_grad():
            self.key_model.eval()
            user_view_2, item_view_2 = self.key_model()
        
        user_cl_loss = self.InfoNCE(user_view_1[u_idx], user_view_2[u_idx], self.temp)
        item_cl_loss = self.InfoNCE(item_view_1[i_idx], item_view_2[i_idx], self.temp)
        
        return user_cl_loss + item_cl_loss

    def InfoNCE(self, query, key, temperature, b_cos=True):
        """
        Computes the InfoNCE (Noise-Contrastive Estimation) loss.

        Args:
            query: The query embeddings.
            key: The key embeddings.
            temperature: The temperature parameter for scaling logits.
            b_cos: A boolean indicating whether to normalize embeddings using cosine similarity.

        Returns:
            The InfoNCE loss.
        """
        if b_cos:
            query = F.normalize(query, dim=1)
            key = F.normalize(key, dim=1)
        
        pos_score = torch.sum(query * key, dim=-1)
        pos_score = torch.exp(pos_score / temperature)
        
        ttl_score = torch.matmul(query, key.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
        
        cl_loss = -torch.log(pos_score / (ttl_score + 1e-6))
        return torch.mean(cl_loss)

    def alignment(self, x, y, alpha=2):
        """
        Calculates the alignment loss between two sets of embeddings.

        Args:
            x: First set of embeddings.
            y: Second set of embeddings.
            alpha: The power to which the L2 norm is raised.

        Returns:
            The alignment loss.
        """
        x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
        return (x - y).norm(p=2, dim=1).pow(alpha).mean()

    def uniformity(self, x, t=2):
        """
        Calculates the uniformity loss of embeddings.

        Args:
            x: Embeddings to calculate uniformity for.
            t: Scaling factor for the exponential.

        Returns:
            The uniformity loss.
        """
        x = F.normalize(x, dim=-1)
        return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()

    def calculate_loss(self, user_emb, item_emb):
        """
        Calculates the alignment and uniformity losses.

        Args:
            user_emb: User embeddings.
            item_emb: Item embeddings.

        Returns:
            A tuple containing:
            - The combined alignment and uniformity loss.
            - The alignment loss.
            - The uniformity loss.
        """
        align = self.alignment(user_emb, item_emb)
        uniform = (self.uniformity(user_emb) + self.uniformity(item_emb)) / 2
        return align + self.gamma * uniform, align, uniform

    def compute_loss(self, users, pos, neg):
        """
        Computes the total loss for TwinCL, including BPR, regularization, alignment, uniformity,
        and momentum contrastive losses.

        Args:
            users: A tensor of user indices.
            pos: A tensor of positive item indices.
            neg: A tensor of negative item indices.

        Returns:
            A dictionary containing:
            - "total_loss": The sum of all loss components.
            - "reg_loss": The regularization loss.
            - "cl_loss": The momentum contrastive loss.
            - "rec_loss": The alignment and uniformity loss.
        """
        (users_emb, pos_emb, neg_emb, 
         userEmb0, posEmb0, negEmb0) = self.get_embedding(users.long(), pos.long(), neg.long())

        # BPR Loss
        pos_scores = torch.sum(torch.mul(users_emb, pos_emb), dim=1)
        neg_scores = torch.sum(torch.mul(users_emb, neg_emb), dim=1)
        bpr_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        
        # Regularization Loss
        reg_loss = (userEmb0.norm(2).pow(2) + posEmb0.norm(2).pow(2) + negEmb0.norm(2).pow(2)) / 2
        reg_loss = reg_loss * self.decay
        
        # Alignment and Uniformity Loss
        rec_loss, align_loss, uniform_loss = self.calculate_loss(users_emb, pos_emb)
        
        # Momentum Contrastive Loss
        cl_loss = self.mcl_rate * self.cal_momentum_loss(users, pos)
        
        total_loss = reg_loss + cl_loss + rec_loss
        
        return {
            "loss": total_loss,
            "reg_loss": reg_loss,
            "cl_loss": cl_loss,
            "rec_loss": rec_loss
        }

    def update_key_encoder(self):
        """
        Updates the parameters of the key encoder using a momentum update rule.
        """
        for param_q, param_k in zip(self.model.parameters(), self.key_model.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    def forward(self, users, items):
        """
        Forward pass for predicting scores of user-item pairs.

        Args:
            users: A tensor of user indices.
            items: A tensor of item indices.

        Returns:
            A tensor of predicted scores for the given user-item pairs.
        """
        all_users, all_items = self.propagate()
        users_emb = all_users[users]
        items_emb = all_items[items]
        return torch.sum(torch.mul(users_emb, items_emb), dim=1)

class TwinCL_Encoder(nn.Module):
    """
    Encoder module for TwinCL, responsible for propagating embeddings through the graph.
    """
    def __init__(self, n_users, n_items, emb_size, n_layers):
        """
        Initializes the TwinCL_Encoder.

        Args:
            n_users: Number of users.
            n_items: Number of items.
            emb_size: Dimensionality of embeddings.
            n_layers: Number of graph convolution layers.
        """
        super(TwinCL_Encoder, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.emb_size = emb_size
        self.n_layers = n_layers
        
        self.user_embedding = nn.Embedding(n_users, emb_size)
        self.item_embedding = nn.Embedding(n_items, emb_size)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        print("--- Use Xavier Uniform initializer ---")
        
        self.sparse_norm_adj = None

    def set_graph(self, graph):
        """
        Sets the sparse normalized adjacency matrix for graph propagation.

        Args:
            graph: A sparse tensor representing the normalized adjacency matrix.
        """
        self.sparse_norm_adj = graph

    def get_ego_embeddings(self):
        """
        Returns the initial (ego) user and item embeddings.

        Returns:
            A tuple containing the user embedding weight matrix and item embedding weight matrix.
        """
        return self.user_embedding.weight, self.item_embedding.weight

    def forward(self):
        """
        Performs the forward pass for graph propagation.

        Returns:
            A tuple containing the aggregated user embeddings and item embeddings after propagation.
        """
        ego_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], 0)
        all_embeddings = [ego_embeddings]
        
        for _ in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            all_embeddings.append(ego_embeddings)
        
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        
        user_all_embeddings, item_all_embeddings = torch.split(
            all_embeddings, [self.n_users, self.n_items])
        
        return user_all_embeddings, item_all_embeddings