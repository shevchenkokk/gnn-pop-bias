
class BasicModel(nn.Module):    
    def __init__(self):
        super(BasicModel, self).__init__()
    
    def get_users_rating(self, users):
        raise NotImplementedError


class LightGCN(BasicModel):
    def __init__(
        self,
        dataset: BasicDataset,
        config: dict,
    ):
        super(LightGCN, self).__init__()
        self.dataset = dataset
        self.config = config
        self.n_users = self.dataset.n_users
        self.n_items = self.dataset.n_items
        self.embedding_dim = self.config["embedding_dim"]
        self.n_layers = self.config["LightGCN_n_layers"]
        self.__init__weight()


    def __init_weight(self):
        """
        Initialize embeddings with normal distribution
        """
        self.user_embs = nn.Embedding(
            num_embeddings=self.n_users, embedding_dim=self.embedding_dim)
        self.item_embs = nn.Embedding(
            num_embeddings=self.n_items, embedding_dim=self.embedding_dim)

        if "pretrain" not in config or not config["pretrain"]:
            nn.init.normal_(self.user_embs.weight, std=0.1)
            nn.init.normal_(self.item_embs.weight, std=0.1)
            print("--- Use normal distribution initializer ---")
        else:
            self.user_embs.weight.data.copy_(torch.from_numpy(self.config["user_embs"]))
            self.item_embs.weight.data.copy_(torch.from_numpy(self.config["item_embs"]))
            print("--- Use pretrained data ---")
        self.graph = self.dataset.get_sparse_graph()
        print("--- LightGCN is ready to go ---")


    def propagate(self) -> tuple:
        """
        Propagate methods for LightGCN
        """
        users_emb = self.user_emb.weight
        items_emb = self.item_emb.weight
        all_emb = torch.cat([users_emb, items_emb])
        
        layer_embeddings = [all_emb]
        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(self.graph, all_emb)
            layer_embeddings.append(all_emb)
        layer_embeddings = torch.stack(layer_embeddings, dim=1)

        out = torch.mean(layer_embeddings, dim=1)
        users, items = torch.split(out, [self.n_users, self.n_items])

    
    def get_users_rating(self, users: torch.tensor) -> torch.tensor:
        """
        Compute item ratings for users
        """
        all_users, all_items = self.propagate()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = torch.matmul(users_emb, items_emb.t())
        return rating


    def get_embedding(
        self,
        users: torch.tensor,
        pos_items: torch.tensor,
        neg_items: torch.tensor
    ) -> tuple:
        all_users, all_items = self.propagate()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.user_embs(users)
        pos_emb_ego = self.item_embs(pos_items)
        neg_emb_ego = self.item_embs(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego


    def bpr_loss(self, users: torch.tensor, pos: torch.tensor, neg: torch.tensor) -> tuple:
        (users_emb, pos_emb, neg_emb, 
        userEmb0, posEmb0, negEmb0) = self.get_embedding(users.long(), pos.long(), neg.long())

        reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) +
                              posEmb0.norm(2).pow(2) +
                              negEmb0.norm(2).pow(2)) / float(len(users))

        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        loss = - (pos_scores - neg_scores).sigmoid().log().mean()


    def forward(self, users: torch.tensor, items: torch.tensor):
        all_users, all_items = self.propagate()

        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_prod = torch.mul(users_emb, items_emb)
        return torch.sum(inner_prod, dim=1).sigmoid()


