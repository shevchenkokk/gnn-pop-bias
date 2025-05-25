class BasicDataset:
    """Base class for all datasets"""
    def __init__(self):
        self.n_users = 0
        self.n_items = 0
        self.train_data = None
        self.test_data = None
        
    def get_sparse_graph(self):
        """Get the sparse adjacency matrix for GNN propagation"""
        raise NotImplementedError