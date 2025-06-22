import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Any, Optional, List
from src.utils.basic_dataset import BasicDataset
from src.models.BasicModel.model import BasicModel


class PopDRL(BasicModel):
    """
    PopDRL (Popularity Debiasing method during Representation Learning) model.
    This model addresses popularity bias in recommender systems by optimizing the
    representation learning process through a combination of LightGCN as a graph encoder,
    Weighted Contrastive Learning (WCL) to mitigate sampling bias, and Popularity Associated Modeling (PAM)
    to augment representations of unpopular items.

    Reference:
    Junsan Zhang, Sini Wu, Te Wang, Fengmei Ding, Jie Zhu.
    "Relieving popularity bias in recommendation via debiasing representation enhancement".
    Complex & Intelligent Systems, 2024. DOI: 10.1007/s40747-024-01649-z.
    """
    def __init__(
        self,
        dataset: BasicDataset,
        config: dict,
    ):
        """
        Initializes the PopDRL model.

        Args:
            dataset (BasicDataset): The dataset object containing user-item interaction data.
            config (dict): A dictionary containing model configuration parameters.
        """
        super(PopDRL, self).__init__()
        self.dataset = dataset
        self.config = config
        self.n_users = self.dataset.n_users
        self.n_items = self.dataset.n_items
        self.embedding_dim = self.config["embedding_dim"]
        self.n_layers = self.config.get("PopDRL_n_layers", 2)
        self.decay = self.config.get("decay", 1e-4)
        self.device = self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        
        # PopDRL specific hyperparameters for multi-task training
        self.tau = self.config.get("tau", 0.2) # Temperature parameter for InfoNCE
        self.lambda_1 = self.config.get("lambda_1", 0.2)  # Weight for Weighted Contrastive Learning (L_wcl)
        self.lambda_2 = self.config.get("lambda_2", 0.05)  # Weight for Popularity-guided Debiasing (L_db)
        self.eta = self.config.get("eta", 0.8) # Threshold for negative validation weighting (η)
        self.K = self.config.get("K", 5) # Size of association set (TopK popular items)
        self.delta = self.config.get("delta", 0.1) # Parameter for noise in embedding augmentation

        self.__init_weight()
        
        # Precompute item popularity and association sets [2]
        self._precompute_popularity_and_associations()

        print("--- PopDRL is ready to go ---")

    def __init_weight(self):
        """
        Initializes the PopDRL_Encoder and sets up the graph.
        """
        self.model = PopDRL_Encoder(self.n_users, self.n_items, self.embedding_dim, self.n_layers)
        self.graph = self.dataset.get_sparse_graph().to(self.device)
        self.model.set_graph(self.graph)
        self.model.to(self.device)

    def _precompute_popularity_and_associations(self):
        """
        Computes item popularity, identifies popular/unpopular items,
        and constructs association sets for unpopular items based on cross-user interactions. [3]
        This method now uses `self.dataset.train_data` from the provided `RecDataset` structure.
        """
        # Calculate item popularity (interaction counts)
        item_counts = torch.zeros(self.n_items, dtype=torch.float32, device=self.device)
        if hasattr(self.dataset, 'train_data'):
            for user_id, items in self.dataset.train_data.items():
                for item_id in items:
                    item_counts[item_id] += 1
            print("--- Item popularity calculated from dataset.train_data ---")
        else:
            print("--- Warning: Item popularity data not found in dataset.train_data. Using uniform dummy popularity. ---")
            item_counts = torch.ones(self.n_items, dtype=torch.float32, device=self.device)
        self.item_popularity = item_counts

        # 2. Define popular and unpopular items
        if self.item_popularity.sum() == 0:
            popularity_threshold_low = 0
            popularity_threshold_high = 0
        else:
            popularity_threshold_low = torch.quantile(self.item_popularity, 0.2).item()
            popularity_threshold_high = torch.quantile(self.item_popularity, 0.8).item()

        self.unpopular_items_mask = (self.item_popularity <= popularity_threshold_low)
        self.popular_items_mask = (self.item_popularity >= popularity_threshold_high)
        
        self.unpopular_item_indices = torch.nonzero(self.unpopular_items_mask).squeeze(1)
        self.popular_item_indices = torch.nonzero(self.popular_items_mask).squeeze(1)

        # 3. Build association sets (A_i) for unpopular items
        self.association_sets_Ai = {}

        if len(self.unpopular_item_indices) > 0 and len(self.popular_item_indices) > 0:
            item_to_users: Dict[int, set] = {i: set() for i in range(self.n_items)}
            if hasattr(self.dataset, 'train_data'):
                for u_id, items in self.dataset.train_data.items():
                    for item_id in items:
                        item_to_users[item_id].add(u_id)

            unpopular_indices_list = self.unpopular_item_indices.tolist()
            popular_indices_list = self.popular_item_indices.tolist()

            for unpop_idx in unpopular_indices_list:
                users_interacted_with_unpop = item_to_users[unpop_idx]

                if not users_interacted_with_unpop:
                    self.association_sets_Ai[unpop_idx] = []
                    continue

                cross_user_counts = {}
                for pop_idx in popular_indices_list:
                    if pop_idx == unpop_idx:
                        continue
                    
                    users_interacted_with_pop = item_to_users[pop_idx]
                    
                    # Calculate intersection size
                    cross_count = len(users_interacted_with_unpop.intersection(users_interacted_with_pop))
                    
                    if cross_count > 0:
                        cross_user_counts[pop_idx] = cross_count
                
                # Sort and take TopK
                sorted_popular_items = sorted(cross_user_counts.items(), key=lambda item: item[1], reverse=True)
                self.association_sets_Ai[unpop_idx] = [item_id for item_id, _ in sorted_popular_items[:self.K]]
        else:
            print("--- Warning: Not enough popular/unpopular items to build association sets. ---")

    def propagate(self) -> Tuple:
        """
        Propagates embeddings through the LightGCN-like encoder. [2]

        Returns:
            Tuple: A tuple containing (user_all_embeddings, item_all_embeddings).
        """
        return self.model()

    def get_users_rating(self, users: torch.Tensor) -> torch.Tensor:
        """
        Generates predicted ratings for a batch of users across all items.

        Args:
            users (torch.Tensor): A tensor of user indices.

        Returns:
            torch.Tensor: A tensor of predicted ratings for the given users across all items.
        """
        all_users, all_items = self.propagate()
        users_emb = all_users[users.long()]
        items_emb = all_items
        return torch.matmul(users_emb, items_emb.t())

    def get_embedding(self, users, pos_items, neg_items):
        """
        Retrieves propagated and ego embeddings for users, positive, and negative items.

        Args:
            users (torch.Tensor): A tensor of user indices.
            pos_items (torch.Tensor): A tensor of positive item indices.
            neg_items (torch.Tensor): A tensor of negative item indices.

        Returns:
            Tuple: A tuple containing:
                                      - users_emb (propagated user embeddings)
                                      - pos_emb (propagated positive item embeddings)
                                      - neg_emb (propagated negative item embeddings)
                                      - userEmb0 (ego user embeddings)
                                      - posEmb0 (ego positive item embeddings)
                                      - negEmb0 (ego negative item embeddings)
        """
        all_users, all_items = self.propagate()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        
        # Get ego embeddings for regularization
        userEmb0, itemEmb0 = self.model.get_ego_embeddings()
        users_emb_ego = userEmb0[users]
        pos_emb_ego = itemEmb0[pos_items]
        neg_emb_ego = itemEmb0[neg_items]
        
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def generate_augmented_views(self, embeddings: torch.Tensor) -> Tuple:
        """
        Generates two augmented views by adding uniformly distributed random noise. [2]
        e'_i = e_i + ε'_i, e''_i = e_i + ε''_i
        where ||ε||_2 = δ, ε = ϵ̄ ⋅ sign(e_i) and ϵ̄ ~ U(0,1)

        Args:
            embeddings (torch.Tensor): The original embeddings to augment.

        Returns:
            Tuple: A tuple containing two augmented views.
        """
        # Generate random noise with uniform distribution U(0,1)
        epsilon_bar_1 = torch.rand_like(embeddings)
        epsilon_bar_2 = torch.rand_like(embeddings)

        # Apply sign(e_i) and normalize to L2-norm δ
        noise_1 = F.normalize(epsilon_bar_1 * torch.sign(embeddings), dim=-1) * self.delta
        noise_2 = F.normalize(epsilon_bar_2 * torch.sign(embeddings), dim=-1) * self.delta
        
        view_1 = embeddings + noise_1
        view_2 = embeddings + noise_2
        return view_1, view_2

    def calculate_negative_validation_weights(self, user_embeddings: torch.Tensor, neg_item_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Calculates the weights ω_j for negative validation. [2]
        ω_j = {0, if s(u,j) > η; 1, if s(u,j) ≤ η}
        s(u,j) = <e_u, e_j> (cosine similarity)

        Args:
            user_embeddings (torch.Tensor): User embeddings.
            neg_item_embeddings (torch.Tensor): Negative item embeddings.

        Returns:
            torch.Tensor: A tensor of weights for the negative items.
        """
        # Normalize for cosine similarity
        user_embeddings_norm = F.normalize(user_embeddings, dim=-1)
        neg_item_embeddings_norm = F.normalize(neg_item_embeddings, dim=-1)
        
        # Compute cosine similarity
        if user_embeddings_norm.dim() == 2 and neg_item_embeddings_norm.dim() == 2:
            similarity_scores = torch.matmul(user_embeddings_norm, neg_item_embeddings_norm.t()) # (num_users_in_batch, num_neg_items_in_batch)
        else:
            # Assuming these are already expanded for pairwise comparison, or are 1D vectors
            similarity_scores = F.cosine_similarity(user_embeddings_norm, neg_item_embeddings_norm, dim=-1)


        # Apply threshold for weights
        weights = torch.ones_like(similarity_scores)
        # Penalize false negatives
        weights[similarity_scores > self.eta] = 0.0
        return weights

    def calculate_wcl_loss(self, users_emb: torch.Tensor, pos_emb: torch.Tensor, neg_emb: torch.Tensor, 
                           users: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
        """
        Calculates the Weighted Contrastive Learning loss (L_wcl = L_1cl + L_2cl). [2]

        Args:
            users_emb (torch.Tensor): Propagated user embeddings.
            pos_emb (torch.Tensor): Propagated positive item embeddings.
            neg_emb (torch.Tensor): Propagated negative item embeddings.
            users (torch.Tensor): User indices.
            pos (torch.Tensor): Positive item indices.
            neg (torch.Tensor): Negative item indices.

        Returns:
            torch.Tensor: The calculated Weighted Contrastive Learning loss.
        """
        # L_1cl: Standard InfoNCE for positive pairs (augmented views)
        # Generate augmented views for users and items
        user_view_1, user_view_2 = self.generate_augmented_views(users_emb)
        item_view_1, item_view_2 = self.generate_augmented_views(pos_emb)

        # InfoNCE for users
        ttl_score_u = torch.matmul(user_view_1, user_view_2.t()) / self.tau
        l1cl_u_loss = -torch.mean(torch.diag(F.log_softmax(ttl_score_u, dim=1)))

        # InfoNCE for items
        ttl_score_i = torch.matmul(item_view_1, item_view_2.t()) / self.tau
        l1cl_i_loss = -torch.mean(torch.diag(F.log_softmax(ttl_score_i, dim=1)))
        
        L1cl = l1cl_u_loss + l1cl_i_loss

        # L_2cl: Weighted InfoNCE for negative pairs
        pos_score_weighted = torch.sum(users_emb * pos_emb, dim=-1) # (batch_size)
        neg_score_weighted = torch.matmul(users_emb, neg_emb.t()) # (batch_size, num_neg_samples)
        
        # Calculate weights for negative items in the batch
        weights_matrix = self.calculate_negative_validation_weights(users_emb.unsqueeze(1), neg_emb.unsqueeze(0))

        # Apply weights to exponential similarity values
        weighted_neg_exp_scores = weights_matrix * torch.exp(neg_score_weighted / self.tau)

        numerator = torch.exp(pos_score_weighted / self.tau)
        # Denominator of InfoNCE for L2cl
        denominator_L2cl = numerator + weighted_neg_exp_scores.sum(dim=1)

        L2cl = -torch.mean(torch.log(numerator / (denominator_L2cl + 1e-8)))
        
        Lwcl = L1cl + L2cl
        return Lwcl

    def calculate_pam_loss(self, all_item_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Calculates the Popularity-guided Debiasing loss (L_db). [2]
        L_db = 1/|G_unpop| ∑_{i∈G_unpop} P_i
        P_i = 1/|A_i| ∑_{j∈A_i} ||(e_i - e_j)||_2

        Args:
            all_item_embeddings (torch.Tensor): All item embeddings from the encoder.

        Returns:
            torch.Tensor: The calculated Popularity-guided Debiasing loss.
        """
        if not self.unpopular_item_indices.numel() > 0:
            return torch.tensor(0.0, device=self.device)

        total_Pi = torch.tensor(0.0, device=self.device)
        num_unpopular_items_with_associations = 0

        # Collect embeddings for unpopular items and their associated popular items
        unpopular_embs_to_process = []
        associated_embs_to_process = []
        association_counts = [] # To store |A_i| for each unpopular item

        # First, gather all necessary embeddings and counts
        for unpop_idx in self.unpopular_item_indices.tolist():
            if unpop_idx not in self.association_sets_Ai or not self.association_sets_Ai[unpop_idx]:
                continue # Skip if no associations

            unpopular_item_associations = self.association_sets_Ai[unpop_idx]
            
            # Get the embedding for the unpopular item
            e_i = all_item_embeddings[unpop_idx]

            # Get embeddings for associated popular items
            e_j_list = [all_item_embeddings[pop_assoc_idx] for pop_assoc_idx in unpopular_item_associations]
            
            if e_j_list: # If there are associations
                unpopular_embs_to_process.append(e_i)
                associated_embs_to_process.append(torch.stack(e_j_list))
                association_counts.append(len(e_j_list))
        
        if not unpopular_embs_to_process:
            return torch.tensor(0.0, device=self.device)

        # Convert lists to tensors for vectorized operations
        unpopular_embs_tensor = torch.stack(unpopular_embs_to_process) # (N_unpop_active, embedding_dim)

        # Calculate P_i for all active unpopular items in a vectorized way        
        final_P_i_values = []
        for i, unpop_idx_val in enumerate(self.unpopular_item_indices.tolist()):
            if unpop_idx_val not in self.association_sets_Ai or not self.association_sets_Ai[unpop_idx_val]:
                continue

            e_i = all_item_embeddings[unpop_idx_val]
            pop_assoc_indices = self.association_sets_Ai[unpop_idx_val]

            # Stack associated item embeddings
            e_j_batch = all_item_embeddings[pop_assoc_indices] # (num_associations, embedding_dim)

            # Calculate difference and norm in a vectorized way for this unpop_idx
            differences = e_i - e_j_batch 
            norms = torch.norm(differences, p=2, dim=-1)

            P_i = torch.mean(norms) # Mean over associations for this unpopular item
            final_P_i_values.append(P_i)

        if final_P_i_values:
            Ldb = torch.mean(torch.stack(final_P_i_values)) # Mean over all active unpopular items
        else:
            Ldb = torch.tensor(0.0, device=self.device)
            
        return Ldb

    def compute_loss(self, users, pos, neg):
        """
        Computes the total loss, combining BPR, Regularization,
        Weighted Contrastive Learning, and Popularity-guided Debiasing. [2]
        L = L_rec + λ_1 * L_wcl + λ_2 * L_db + λ_3 * ||Θ||_2

        Args:
            users (torch.Tensor): User indices.
            pos (torch.Tensor): Positive item indices.
            neg (torch.Tensor): Negative item indices.

        Returns:
            Dict: A dictionary containing various loss components:
                                     - "total_loss"
                                     - "bpr_loss"
                                     - "reg_loss"
                                     - "wcl_loss"
                                     - "pam_loss"
        """
        all_users, all_items = self.propagate()

        users_emb = all_users[users.long()]
        pos_emb = all_items[pos.long()]
        neg_emb = all_items[neg.long()]
        
        userEmb0, itemEmb0 = self.model.get_ego_embeddings()
        users_emb_ego = userEmb0[users.long()]
        pos_emb_ego = itemEmb0[pos.long()]
        neg_emb_ego = itemEmb0[neg.long()]

        # Bayesian Personalized Ranking (BPR) Loss
        pos_scores = torch.sum(torch.mul(users_emb, pos_emb), dim=1)
        neg_scores = torch.sum(torch.mul(users_emb, neg_emb), dim=1)
        bpr_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))

        # Regularization Loss (L2 regularization on ego embeddings)
        reg_loss = (users_emb_ego.norm(2).pow(2) + pos_emb_ego.norm(2).pow(2) + neg_emb_ego.norm(2).pow(2)) / 2
        reg_loss = reg_loss * self.decay

        # Weighted Contrastive Learning Loss (L_wcl)
        wcl_loss = self.calculate_wcl_loss(users_emb, pos_emb, neg_emb, users, pos, neg)

        # Popularity-guided Debiasing Loss (L_db)
        pam_loss = self.calculate_pam_loss(all_items)

        # Total Loss
        total_loss = bpr_loss + self.lambda_1 * wcl_loss + self.lambda_2 * pam_loss + reg_loss

        return {
            "loss": total_loss,
            "bpr_loss": bpr_loss,
            "reg_loss": reg_loss,
            "wcl_loss": wcl_loss,
            "pam_loss": pam_loss
        }

    def forward(self, users, items):
        """
        Predicts scores for specific user-item pairs.

        Args:
            users (torch.Tensor): A tensor of user indices.
            items (torch.Tensor): A tensor of item indices.

        Returns:
            torch.Tensor: A tensor of predicted scores for the given user-item pairs.
        """
        all_users, all_items = self.propagate()
        users_emb = all_users[users]
        items_emb = all_items[items]
        return torch.sum(torch.mul(users_emb, items_emb), dim=1)

class PopDRL_Encoder(nn.Module):
    """
    PopDRL_Encoder implements a LightGCN-like graph neural network
    for learning user and item embeddings. [5, 6, 2]
    """
    def __init__(self, n_users, n_items, emb_size, n_layers):
        """
        Initializes the PopDRL_Encoder.

        Args:
            n_users (int): Number of unique users.
            n_items (int): Number of unique items.
            emb_size (int): Dimensionality of user and item embeddings.
            n_layers (int): Number of graph convolution layers.
        """
        super(PopDRL_Encoder, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.emb_size = emb_size
        self.n_layers = n_layers
        
        self.user_embedding = nn.Embedding(n_users, emb_size)
        self.item_embedding = nn.Embedding(n_items, emb_size)

        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        print("--- Using Xavier Uniform initializer ---")
        
        self.sparse_norm_adj = None

    def set_graph(self, graph):
        """
        Sets the normalized adjacency matrix for graph propagation. [2]

        Args:
            graph (torch.Tensor): The sparse normalized adjacency matrix.
        """
        self.sparse_norm_adj = graph

    def get_ego_embeddings(self) -> Tuple:
        """
        Returns the initial (ego) embeddings of users and items. [2]

        Returns:
            Tuple: A tuple containing
                                               (user_embedding_weights, item_embedding_weights).
        """
        return self.user_embedding.weight, self.item_embedding.weight

    def forward(self) -> Tuple:
        """
        Performs LightGCN-like graph convolution and aggregates embeddings. [2]

        Returns:
            Tuple: A tuple containing
                                               (user_all_embeddings, item_all_embeddings).
        """
        # Concatenate user and item ego embeddings
        ego_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], 0)
        all_embeddings = [ego_embeddings]
        
        # Propagate embeddings through layers
        for _ in range(self.n_layers):
            # LightGCN performs sparse matrix multiplication without non-linearities
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            all_embeddings.append(ego_embeddings)
        
        # Aggregate embeddings from all layers by mean-pooling
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(
            all_embeddings, [self.n_users, self.n_items])
        
        return user_all_embeddings, item_all_embeddings