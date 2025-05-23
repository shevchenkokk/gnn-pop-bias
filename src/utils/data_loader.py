import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Set
from torch.utils.data import Dataset

class RecDataset(Dataset):
    """
    Dataset class for recommendation data
    """
    def __init__(self, data_path, dataset_name, split_ratio=0.8, seed=42):
        """
        Initialize dataset
        
        Args:
            data_path: Path to dataset directory
            dataset_name: Name of the dataset
            split_ratio: Train-test split ratio
            seed: Random seed for reproducibility
        """
        self.data_path = Path(data_path)
        self.dataset_name = dataset_name
        self.split_ratio = split_ratio
        self.seed = seed
        
        # Load and preprocess data
        print(f"--- Loading data for {dataset_name}... ---")
        self.user_item_dict = self._load_data()
        print("--- Data loaded. Splitting data... ---")
        self.train_data, self.test_data = self._split_data()
        
        # Get user and item counts
        all_users = set(self.user_item_dict.keys())
        all_items = set()
        for items in self.user_item_dict.values():
            all_items.update(items)

        self.n_users = max(all_users) + 1 if all_users else 0
        self.n_items = max(all_items) + 1 if all_items else 0

        print(f"--- Number of users: {self.n_users}, Number of items: {self.n_items} ---")
        
        # Create user-item interaction matrix
        print("--- Creating interaction matrix... ---")
        self.interaction_matrix = self._create_interaction_matrix()
        
        # Create training samples
        print("--- Creating training samples... ---")
        self.train_users, self.train_pos_items = self._create_train_samples()
        print("--- Dataset initialization complete ---")
        
    def _load_data(self) -> Dict[int, List[int]]:
        """
        Load dataset from file
        
        Returns:
            Dictionary mapping user IDs to lists of item IDs
        """
        user_item_dict = {}
        
        if self.dataset_name == 'ml-32m':
            # Load MovieLens data
            ratings_file = self.data_path / self.dataset_name / 'ratings.csv'
            df = pd.read_csv(ratings_file)
            
            # Filter positive interactions (rating >= 4)
            df = df[df['rating'] >= 4]
            
            # Create user-item dictionary
            for user, items in df.groupby('userId')['movieId']:
                user_item_dict[user] = items.tolist()
                
        elif self.dataset_name == 'yelp2018':
            # Load Yelp data
            data_file = self.data_path / self.dataset_name / 'yelp2018.inter'
            df = pd.read_csv(data_file, sep=' ')
            
            # Create user-item dictionary
            for user, items in df.groupby('user_id')['item_id']:
                user_item_dict[user] = items.tolist()
                
        elif self.dataset_name == 'lastfm':
            # Load LastFM data
            data_file = self.data_path / self.dataset_name / 'user_artists.dat'
            df = pd.read_csv(data_file, sep='\t')
            
            # Create user-item dictionary
            for user, items in df.groupby('userID')['artistID']:
                user_item_dict[user] = items.tolist()
                
        elif self.dataset_name == 'gowalla':
            # Load Gowalla data
            data_file = self.data_path / self.dataset_name / 'gowalla.txt'
            df = pd.read_csv(data_file, sep='\t', header=None, names=['user', 'time', 'lat', 'lon', 'item'])
            
            # Create user-item dictionary
            for user, items in df.groupby('user')['item']:
                user_item_dict[user] = items.tolist()
                
        elif self.dataset_name == 'amazon-book':
            # Load Amazon Book data
            data_file = self.data_path / self.dataset_name / 'Books.csv'
            df = pd.read_csv(data_file)
            
            # Create user-item dictionary
            for user, items in df.groupby('reviewerID')['asin']:
                user_item_dict[user] = items.tolist()
                
        # Remap IDs to consecutive integers
        user_map = {u: i for i, u in enumerate(user_item_dict.keys())}
        item_set = set()
        for items in user_item_dict.values():
            item_set.update(items)
        item_map = {i: j for j, i in enumerate(item_set)}
        
        # Create new dictionary with remapped IDs
        new_dict = {}
        for user, items in user_item_dict.items():
            new_dict[user_map[user]] = [item_map[item] for item in items]
            
        return new_dict
        
    def _split_data(self) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
        """
        Split data into training and testing sets
        
        Returns:
            Tuple of (train_data, test_data) dictionaries
        """
        train_data = {}
        test_data = {}
        
        # Set random seed for reproducibility
        np.random.seed(self.seed)
        
        for user, items in self.user_item_dict.items():
            items = list(set(items))  # Remove duplicates
            
            if len(items) < 2:
                train_data[user] = items
                continue
                
            # Shuffle items
            np.random.shuffle(items)
            
            # Split based on ratio
            train_size = int(len(items) * self.split_ratio)
            train_data[user] = items[:train_size]
            test_data[user] = items[train_size:]
            
        return train_data, test_data
        
    def _create_interaction_matrix(self) -> torch.sparse.FloatTensor:
        """
        Create sparse interaction matrix
        
        Returns:
            Sparse tensor of user-item interactions
        """
        rows = []
        cols = []
        
        for user, items in self.train_data.items():
            for item in items:
                rows.append(user)
                cols.append(item)
                
        values = torch.ones(len(rows))
        indices = torch.LongTensor([rows, cols])
        
        return torch.sparse_coo_tensor(
            indices, 
            values, 
            torch.Size([self.n_users, self.n_items]),
            dtype=torch.float
        )
        
    def _create_train_samples(self) -> Tuple[List[int], List[int]]:
        """
        Create training samples
        
        Returns:
            Tuple of (users, positive_items) lists
        """
        users = []
        pos_items = []
        
        for user, items in self.train_data.items():
            for item in items:
                users.append(user)
                pos_items.append(item)
                
        return users, pos_items
        
    def __len__(self):
        """Get dataset length"""
        return len(self.train_users)
        
    def __getitem__(self, idx):
        """
        Get training sample
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (user, positive_item, negative_item)
        """
        user = self.train_users[idx]
        pos_item = self.train_pos_items[idx]
        
        # Sample negative item
        while True:
            neg_item = np.random.randint(0, self.n_items)
            if neg_item not in self.user_item_dict.get(user, set()):
                break
                
        return user, pos_item, neg_item
        
    def get_sparse_graph(self) -> torch.sparse.FloatTensor:
        num_users = self.n_users
        num_items = self.n_items
        num_total_nodes = num_users + num_items

        if num_users == 0 or num_items == 0:
            return torch.sparse_coo_tensor(
                torch.LongTensor([[], []]),
                torch.FloatTensor([]),
                torch.Size([num_total_nodes, num_total_nodes]),
                dtype=torch.float
            )

        R_indices = self.interaction_matrix._indices()
        R_values = self.interaction_matrix._values()

        adj_tilde_row_indices_R = R_indices[0]
        adj_tilde_col_indices_R = R_indices[1] + num_users

        adj_tilde_row_indices_RT = R_indices[1] + num_users
        adj_tilde_col_indices_RT = R_indices[0]

        A_tilde_row_indices = torch.cat([adj_tilde_row_indices_R, adj_tilde_row_indices_RT])
        A_tilde_col_indices = torch.cat([adj_tilde_col_indices_R, adj_tilde_col_indices_RT])
        A_tilde_values = torch.cat([R_values, R_values.clone()])

        A_tilde_indices_tensor = torch.stack([A_tilde_row_indices, A_tilde_col_indices])
        
        eye_indices = torch.stack([
            torch.arange(num_total_nodes, dtype=torch.long),
            torch.arange(num_total_nodes, dtype=torch.long)
        ])
        eye_values = torch.ones(num_total_nodes, dtype=torch.float)

        A_hat_indices = torch.cat([A_tilde_indices_tensor, eye_indices], dim=1)
        A_hat_values = torch.cat([A_tilde_values, eye_values])
        
        A_hat = torch.sparse_coo_tensor(
            A_hat_indices,
            A_hat_values,
            torch.Size([num_total_nodes, num_total_nodes]),
            dtype=torch.float
        ).coalesce()

        degrees_hat = torch.sparse.sum(A_hat, dim=1).to_dense().squeeze()
        degrees_hat[degrees_hat == 0] = 1e-12 

        d_hat_inv_sqrt = torch.pow(degrees_hat, -0.5)
        d_hat_inv_sqrt[torch.isinf(d_hat_inv_sqrt)] = 0.
        d_hat_inv_sqrt[torch.isnan(d_hat_inv_sqrt)] = 0.

        A_hat_final_indices = A_hat._indices()
        A_hat_final_values = A_hat._values()
        
        row_indices = A_hat_final_indices[0]
        col_indices = A_hat_final_indices[1]
        
        normalized_values_tensor = A_hat_final_values * \
                           d_hat_inv_sqrt[row_indices] * \
                           d_hat_inv_sqrt[col_indices]
        
        S_matrix = torch.sparse_coo_tensor(
            A_hat_final_indices,
            normalized_values_tensor,
            torch.Size([num_total_nodes, num_total_nodes]),
            dtype=torch.float
        )

        return S_matrix.coalesce()