import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Set, Literal
from torch.utils.data import Dataset
from datasets import DatasetDict, load_dataset


class RecDataset(Dataset):
    """
    Dataset class for recommendation data
    """
    def __init__(self,
                 data_path,
                 dataset_name,
                 n_candidate_negs=1,
                 split_ratio=0.8,
                 k_core=10,
                 apply_ips_cn=False,
                 seed=42):
        """
        Initialize dataset
        
        Args:
            data_path: Path to dataset directory
            dataset_name: Name of the dataset
            split_ratio: Train-test split ratio
            k_core: Minimum number of interactions for users and items
            seed: Random seed for reproducibility
        """
        self.data_path = Path(data_path)
        self.dataset_name = dataset_name
        self.n_candidate_negs = n_candidate_negs
        self.split_ratio = split_ratio
        self.k_core = k_core
        self.apply_ips_cn = apply_ips_cn
        self.seed = seed

        # Load and preprocess data
        print(f"--- Loading data for {dataset_name}... ---")
        raw_df_interactions = self._load_data()

        if self.k_core and self.k_core > 1:
            print(f"--- Applying {self.k_core}-core filtering... ---")
            print(f"--- Interactions before filtering: {len(raw_df_interactions)} ---")
            raw_df_interactions = self._k_core_filtering(raw_df_interactions, self.k_core)
            print(f"--- Filtering done. Final interactions: {len(raw_df_interactions)} ---")

        print("--- Data loaded. Splitting data... ---")
        self.train_data, self.test_data = self._split_data(raw_df_interactions)

        # Get user and item counts
        all_users = set(self.train_data['uid'].unique())
        all_users.update(self.test_data['uid'].unique())
        
        all_items = set(self.train_data['item_id'].unique())
        all_items.update(self.test_data['item_id'].unique())

        self.user_mapping = {u: i for i, u in enumerate(sorted(list(all_users)))}
        self.item_mapping = {i: j for j, i in enumerate(sorted(list(all_items)))}

        self.n_users = len(self.user_mapping)
        self.n_items = len(self.item_mapping)

        print(f"--- Number of users: {self.n_users}, Number of items: {self.n_items} ---")

        self.train_data['uid'] = self.train_data['uid'].map(self.user_mapping)
        self.train_data['item_id'] = self.train_data['item_id'].map(self.item_mapping)
        self.test_data['uid'] = self.test_data['uid'].map(self.user_mapping)
        self.test_data['item_id'] = self.test_data['item_id'].map(self.item_mapping)

        self.train_data = self.train_data.groupby('uid')['item_id'].apply(list).to_dict()
        self.test_data = self.test_data.groupby('uid')['item_id'].apply(list).to_dict()

        # Create user-item interaction matrix
        print("--- Creating interaction matrix... ---")
        self.interaction_matrix = self._create_interaction_matrix()

        if self.apply_ips_cn:
            print("--- Computing IPS-CN weights... ---")
            self._calculate_propensity_scores()
            self._calculate_ips_cn_weights()
            print("--- IPS-CN weights computed ---")
        else:
            self.item_popularity = None
            self.ips_cn_weights = None

        self.num_train_interactions = sum(len(items) for items in self.train_data.values())
        self.num_test_interactions = sum(len(items) for items in self.test_data.values())

        print(f"--- Number of train interactions: {self.num_train_interactions} ---")
        print(f"--- Number of test interactions: {self.num_test_interactions} ---")

        # Create training samples
        print("--- Creating training samples... ---")
        self.train_users, self.train_pos_items = self._create_train_samples()

        self.all_item_ids = list(range(self.n_items))
        print("--- Dataset initialization complete ---")

    def _k_core_filtering(self, df: pd.DataFrame, k: int) -> pd.DataFrame:
        """
        Apply k-core filtering to the dataframe.
        
        Args:
            df: DataFrame with 'uid', 'item_id' columns.
            k: The minimum number of interactions for each user and item.
            
        Returns:
            Filtered DataFrame.
        """
        while True:
            initial_interactions = len(df)

            # Filter by users
            user_counts = df['uid'].value_counts()
            valid_users = user_counts[user_counts >= k].index
            df = df[df['uid'].isin(valid_users)]
            
            # Filter by items
            item_counts = df['item_id'].value_counts()
            valid_items = item_counts[item_counts >= k].index
            df = df[df['item_id'].isin(valid_items)]
            
            final_interactions = len(df)
            
            if initial_interactions == final_interactions:
                break
                
        return df
        
    def _load_data(self) -> pd.DataFrame:
        """
        Load dataset from file. Always returns a DataFrame with 'uid', 'item_id', 'timestamp'.
        
        Returns:
            DataFrame of interactions.
        """
        user_item_dict = {}
        
        if self.dataset_name == 'ml-1m':
            ratings_file = self.data_path / self.dataset_name / 'ratings.dat'

            df = pd.read_csv(ratings_file, sep='::', engine='python',
                             names=['uid', 'item_id', 'rating', 'timestamp'])

            df_interactions = df[['uid', 'item_id', 'timestamp']].sort_values(by='timestamp').reset_index(drop=True)

        elif self.dataset_name == 'yambda-50m':
            yambda_loader = YambdaDataset("flat", "50m")

            all_interactions_df_parts = []
            print("--- Loading 'likes' interactions... ---")
            likes_hf_dataset = yambda_loader.interaction("likes")
            likes_df = likes_hf_dataset.to_pandas()
            # Filter only organic likes
            likes_df = likes_df[likes_df['is_organic'] == True]
            all_interactions_df_parts.append(likes_df[['uid', 'item_id', 'timestamp']])

            print("--- Loading 'listens' interactions... ---")
            listens_hf_dataset = yambda_loader.interaction("listens")
            listens_df = listens_hf_dataset.to_pandas()
            # Filter organic listens with high played ratio
            listens_df = listens_df[(listens_df['is_organic'] == True) & (listens_df['played_ratio_pct'] >= 80)]
            all_interactions_df_parts.append(listens_df[['uid', 'item_id', 'timestamp']])

            df_interactions = pd.concat(all_interactions_df_parts).drop_duplicates().reset_index(drop=True)
            df_interactions = df_interactions.sort_values(by='timestamp').reset_index(drop=True)

        elif self.dataset_name == "kion":
            interactions_file = self.data_path / self.dataset_name / 'data_en' / 'interactions.csv'

            df_interactions = pd.read_csv(interactions_file)

            df_interactions.rename(columns={
                'user_id': 'uid',
                'last_watch_dt': 'timestamp'
            }, inplace=True)
            
            df_interactions['timestamp'] = pd.to_datetime(df_interactions['timestamp'])
            df_interactions['timestamp'] = df_interactions['timestamp'].astype(np.int64) // 10**9

            df_interactions = df_interactions[df_interactions['watched_pct'] >= 50.0].copy()

            df_interactions = df_interactions.sort_values(by='timestamp').reset_index(drop=True)
            df_interactions = df_interactions[['uid', 'item_id', 'timestamp']]

        elif self.dataset_name == 'yelp2021':
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

        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
            
        return df_interactions
        
    def _split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into training and testing sets based on Global Temporal Split
        
        Args:
            df: DataFrame with 'uid', 'item_id', 'timestamp' columns, already sorted by 'timestamp'.
            
        Returns:
            Tuple of (train_df, test_df) DataFrames.
        """
        one_day_in_seconds = 1 * 24 * 60 * 60
        thirty_minutes_in_seconds = 30 * 60

        if self.dataset_name == 'yambda-50m':
            test_duration_seconds = one_day_in_seconds
        elif self.dataset_name == 'ml-1m':
            test_duration_seconds = 30 * one_day_in_seconds
        elif self.dataset_name == 'kion':
            test_duration_seconds = 7 * one_day_in_seconds

        max_timestamp = df['timestamp'].max()

        test_start_timestamp = max_timestamp - test_duration_seconds
        gap_start_timestamp = test_start_timestamp - thirty_minutes_in_seconds

        train_df = df[df['timestamp'] < gap_start_timestamp].copy()
        test_df = df[df['timestamp'] >= test_start_timestamp].copy()

        return train_df, test_df

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

        self.n_interactions = len(rows)
                
        values = torch.ones(len(rows))
        indices = torch.LongTensor([rows, cols])
        
        return torch.sparse_coo_tensor(
            indices, 
            values,
            torch.Size([self.n_users, self.n_items]),
            dtype=torch.float
        )
    
    def _compute_item_popularity(self) -> np.ndarray:
        """
        Compute popularity for each item (number of interactions)
        
        Returns:
            NumPy array where index is item_id and value is its popularity
        """
        item_counts = np.zeros(self.n_items, dtype=np.int32)
        for user_id, item_ids in self.train_data.items():
            for item_id in item_ids:
                item_counts[item_id] += 1
        return item_counts
    
    def _calculate_propensity_scores(self) -> np.ndarray:
        self.item_popularity = self._compute_item_popularity()
        total_interactions = self.item_popularity.sum()
        item_prob = self.item_popularity / total_interactions

        ips_alpha = 0.5
        propensity_scores = np.power(item_prob, ips_alpha)
        self.propensity_scores = torch.FloatTensor(propensity_scores)

    def _calculate_ips_cn_weights(self):
        ips_cn_weights = 1 / (self.propensity_scores + 1e-8)
        
        train_items_mask = torch.from_numpy(self.item_popularity > 0).to(ips_cn_weights.device)
        mean_of_train_weights = ips_cn_weights[train_items_mask].mean()
        ips_cn_weights /= mean_of_train_weights

        ips_cn_weights[~train_items_mask] = 1.0

        self.ips_cn_weights = ips_cn_weights

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
        interacted_items = set(self.train_data.get(user, []))

        neg_items_for_sample = []
        while len(neg_items_for_sample) < self.n_candidate_negs:
            neg_item = np.random.randint(0, self.n_items)
            if neg_item not in interacted_items and neg_item not in neg_items_for_sample:
                neg_items_for_sample.append(neg_item)
        if len(neg_items_for_sample) == 1:
            return user, pos_item, *neg_items_for_sample
        else:
            return user, pos_item, np.array(neg_items_for_sample, dtype=np.int64)

    def get_sparse_graph(self) -> torch.sparse.FloatTensor:
        """
        Creates and returns the normalized adjacency matrix.
        This implementation follows the standard LightGCN formulation: S = D^(-1/2) * A * D^(-1/2),
        
        Returns:
            A sparse torch.FloatTensor representing the normalized graph.
        """
        print("--- Creating normalized adjacency matrix... ---")
        
        num_users = self.n_users
        num_items = self.n_items
        num_total_nodes = num_users + num_items

        # Check for empty graph
        if num_users == 0 or num_items == 0 or self.interaction_matrix._nnz() == 0:
            return torch.sparse_coo_tensor(
                torch.LongTensor([[], []]),
                torch.FloatTensor([]),
                torch.Size([num_total_nodes, num_total_nodes]),
                dtype=torch.float
            )

        R_indices = self.interaction_matrix._indices()
        
        R_values = torch.ones_like(self.interaction_matrix._values(), dtype=torch.float)

        # Indices for upper-right block (user-item)
        adj_row_R = R_indices[0]
        adj_col_R = R_indices[1] + num_users

        # Indices for lower-left block (item-user)
        adj_row_RT = R_indices[1] + num_users
        adj_col_RT = R_indices[0]

        # Collect indexes and values for the entire matrix A
        adj_row = torch.cat([adj_row_R, adj_row_RT])
        adj_col = torch.cat([adj_col_R, adj_col_RT])
        adj_values = torch.cat([R_values, R_values.clone()]) # Clone values for symmetrical part
        
        adj_indices = torch.stack([adj_row, adj_col])
        
        A = torch.sparse_coo_tensor(
            adj_indices,
            adj_values,
            torch.Size([num_total_nodes, num_total_nodes]),
            dtype=torch.float
        ).coalesce() # .coalesce() summarizes duplicates, if any

        # Calculate the degrees of the nodes D
        # torch.sparse.sum returns a sparse tensor, convert it to a dense one
        degrees = torch.sparse.sum(A, dim=1).to_dense()

        # Calculate D^(-1/2)
        # Add a small number for stability to avoid division by zero
        degrees[degrees == 0] = 1e-12 
        d_inv_sqrt = torch.pow(degrees, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        
        A_indices = A._indices()
        A_values = A._values()
        
        # Get the degree vectors for the rows and columns of each edge
        d_inv_sqrt_rows = d_inv_sqrt[A_indices[0]]
        d_inv_sqrt_cols = d_inv_sqrt[A_indices[1]]
        
        # Calculating normalized values
        normalized_values = A_values * d_inv_sqrt_rows * d_inv_sqrt_cols
        
        # Creating the final sparse matrix S
        S_matrix = torch.sparse_coo_tensor(
            A_indices,
            normalized_values,
            torch.Size([num_total_nodes, num_total_nodes]),
            dtype=torch.float
        )
        
        print("--- Normalized adjacency matrix created successfully. ---")
        return S_matrix.coalesce()


class YambdaDataset:
    INTERACTIONS = frozenset([
        "likes", "listens", "multi_event", "dislikes", "unlikes", "undislikes"
    ])

    def __init__(
        self,
        dataset_type: Literal["flat", "sequential"] = "flat",
        dataset_size: Literal["50m", "500m", "5b"] = "50m"
    ):
        assert dataset_type in {"flat", "sequential"}
        assert dataset_size in {"50m", "500m", "5b"}
        self.dataset_type = dataset_type
        self.dataset_size = dataset_size

    def interaction(self, event_type: Literal[
        "likes", "listens", "multi_event", "dislikes", "unlikes", "undislikes"
    ]) -> Dataset:
        assert event_type in YambdaDataset.INTERACTIONS
        return self._download(f"{self.dataset_type}/{self.dataset_size}", event_type)

    def audio_embeddings(self) -> Dataset:
        return self._download("", "embeddings")

    def album_item_mapping(self) -> Dataset:
        return self._download("", "album_item_mapping")

    def artist_item_mapping(self) -> Dataset:
        return self._download("", "artist_item_mapping")

    @staticmethod
    def _download(data_dir: str, file: str) -> Dataset:
        data = load_dataset("yandex/yambda", data_dir=data_dir, data_files=f"{file}.parquet")
        # Returns DatasetDict; extracting the only split
        assert isinstance(data, DatasetDict)
        return data["train"]
