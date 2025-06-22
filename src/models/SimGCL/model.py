import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict
from src.utils.basic_dataset import BasicDataset
from src.models.BasicModel.model import BasicModel


class SimGCL(BasicModel):
    """
    SimGCL: Are Graph Augmentations Necessary? Simple Graph Contrastive Learning for Recommendation
    
    Reference: 
    Yu, J., Yin, H., Gao, M., Xia, X., Zhang, X., & Vucetic, S. (2022). 
    Are Graph Augmentations Necessary? Simple Graph Contrastive Learning for Recommendation. 
    SIGIR 2022.
    
    This implementation has been corrected to align with the original paper's methodology.
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
        Initialize embeddings.
        Original paper uses xavier_uniform, so we stick to it.
        """
        self.user_embs = nn.Embedding(
            num_embeddings=self.n_users, embedding_dim=self.embedding_dim)
        self.item_embs = nn.Embedding(
            num_embeddings=self.n_items, embedding_dim=self.embedding_dim)

        # The official code uses xavier_uniform_
        nn.init.xavier_uniform_(self.user_embs.weight)
        nn.init.xavier_uniform_(self.item_embs.weight)
        print("--- Use Xavier Uniform initializer ---")
        
        # Get the graph adjacency matrix in sparse format
        self.graph = self.dataset.get_sparse_graph().to(self.device)
        print("--- SimGCL is ready to go ---")
        
    def _add_noise_to_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Add uniform noise to embeddings for augmentation.
        This is the core of SimGCL's "Simple Graph Contrastive Learning".
        
        Args:
            embeddings: Input embeddings
            
        Returns:
            Augmented embeddings with noise
        """
        random_noise = torch.rand_like(embeddings, device=self.device)
        # The noise is added proportionally to the embedding magnitude
        noise = torch.sign(embeddings) * F.normalize(random_noise, dim=-1) * self.noise_scale
        return embeddings + noise

    def propagate(self, add_noise: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Propagate embeddings through the graph layers.
        
        CORRECTION 1: The original SimGCL does NOT include the initial embeddings (E_0)
        in the final layer aggregation. It only aggregates embeddings from layer 1 to L.
        
        Args:
            add_noise: Whether to add noise during propagation for augmentation.
            
        Returns:
            Tuple of final user and item embeddings.
        """
        users_emb_initial = self.user_embs.weight
        items_emb_initial = self.item_embs.weight
        all_emb = torch.cat([users_emb_initial, items_emb_initial])
        
        layer_embs = [] # Will store embeddings from layer 1 to L
        
        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(self.graph, all_emb)
            if add_noise:
                all_emb = self._add_noise_to_embeddings(all_emb)
            layer_embs.append(all_emb)
        
        # Mean aggregation of propagated layers ONLY (L_1 to L_n)
        final_embs = torch.mean(torch.stack(layer_embs, dim=1), dim=1)
        
        users, items = torch.split(final_embs, [self.n_users, self.n_items])
        return users, items
    
    def _contrastive_loss(self, view1: torch.Tensor, view2: torch.Tensor) -> torch.Tensor:
        """
        Compute InfoNCE contrastive loss between two views.
        
        Args:
            view1: Embeddings from the first augmented view.
            view2: Embeddings from the second augmented view.
            
        Returns:
            The contrastive loss.
        """
        view1 = F.normalize(view1, p=2, dim=1)
        view2 = F.normalize(view2, p=2, dim=1)
        
        # Positive pairs: dot product of corresponding embeddings in the two views
        pos_scores = torch.sum(view1 * view2, dim=1)
        # All pairs: matrix multiplication between the two views
        all_scores = torch.matmul(view1, view2.T)
        
        # InfoNCE loss calculation
        loss = -torch.log(torch.exp(pos_scores / self.temp) / torch.exp(all_scores / self.temp).sum(dim=1))
        
        return loss.mean()
    
    def compute_loss(self, users: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute the total loss for SimGCL, combining BPR and Contrastive Loss.
        
        CORRECTION 2: The entire computation logic is restructured to avoid conflicting gradients.
        We generate two augmented views FIRST. One is used for the BPR loss, and both are used for the CL loss.
        
        Args:
            users: User indices tensor from the batch.
            pos: Positive item indices tensor.
            neg: Negative item indices tensor.

        Returns:
            A dictionary containing total_loss, bpr_loss, reg_loss, and cl_loss.
        """
        # --- Step 1: Generate two augmented views ---
        # Each call to propagate creates a different random noise pattern.
        users_v1, items_v1 = self.propagate(add_noise=True)
        users_v2, items_v2 = self.propagate(add_noise=True)
        
        # --- Step 2: Compute BPR loss using the first view (v1) ---
        # This ensures the recommendation task is learned on an augmented view,
        # making the model robust.
        users_emb_v1 = users_v1[users]
        pos_emb_v1 = items_v1[pos]
        neg_emb_v1 = items_v1[neg]
        
        pos_scores = torch.sum(users_emb_v1 * pos_emb_v1, dim=1)
        neg_scores = torch.sum(users_emb_v1 * neg_emb_v1, dim=1)
        
        bpr_loss = -F.logsigmoid(pos_scores - neg_scores).mean()
        
        # --- Step 3: Compute Regularization Loss on initial embeddings ---
        # The regularization is applied to the original, trainable embeddings (E_0), not the propagated ones.
        user_emb0 = self.user_embs(users)
        pos_emb0 = self.item_embs(pos)
        neg_emb0 = self.item_embs(neg)
        
        reg_loss = (user_emb0.norm(2).pow(2) + 
                    pos_emb0.norm(2).pow(2) + 
                    neg_emb0.norm(2).pow(2)) / (2 * len(users))

        # --- Step 4: Compute Contrastive Loss between the two views (v1 and v2) ---
        # We compute CL loss only for the unique users and items present in the current batch
        # to save computation, as done in the original implementation.
        sampled_users = users.unique()
        sampled_items = torch.cat([pos, neg]).unique()
        
        u_cl_loss = self._contrastive_loss(users_v1[sampled_users], users_v2[sampled_users])
        i_cl_loss = self._contrastive_loss(items_v1[sampled_items], items_v2[sampled_items])
        cl_loss = u_cl_loss + i_cl_loss

        # --- Step 5: Combine all losses ---
        total_loss = bpr_loss + self.decay * reg_loss + self.cl_rate * cl_loss

        return {
            "loss": total_loss,
            "bpr_loss": bpr_loss,
            "reg_loss": reg_loss,
            "cl_loss": cl_loss
        }
    
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

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        all_users, all_items = self.propagate(add_noise=False)
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_prod = torch.sum(users_emb * items_emb, dim=1)
        return inner_prod