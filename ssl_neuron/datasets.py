import torch
import pickle
import numpy as np
from tqdm import tqdm
from pathlib import Path
from torch.multiprocessing import Manager
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from ssl_neuron.utils import subsample_graph, rotate_graph, jitter_node_pos, translate_soma_pos, scale_hull, get_leaf_branch_nodes, compute_node_distances, drop_random_branch, remap_neighbors, neighbors_to_adjacency_torch, adjacency_to_neighbors, BalancedSampler
from torch.utils.data import WeightedRandomSampler, Sampler

class GraphDataset(Dataset):
    """ Dataset of neuronal graphs.

    Neuronal graphs are assumed to be soma-centered (i.e. soma node
    position is (0, 0, 0) and axons have been removed. Node positions
    are assumed to be in microns and y-axis is orthogonal to the pia.
    """
    def __init__(self, config, mode='train', inference=False):

        self.config = config
        self.mode = mode
        self.inference = inference
        data_path = config['data']['path']

        # Augmentation parameters.
        self.jitter_var = config['data']['jitter_var']
        self.rotation_axis = config['data']['rotation_axis']
        self.n_drop_branch = config['data']['n_drop_branch']
        self.translate_var = config['data']['translate_var']
        self.hull_scale_range = config['data'].get('hull_scale_range', (0.0, 0.0))
        self.n_nodes = config['data']['n_nodes']

        data_path = config['data']['path']

        cell_meta = pd.read_csv(Path(data_path, 'merged_features.csv'))

        subtype_by_cell_id = dict(
            zip(cell_meta['cell'].astype(np.str_), cell_meta['celltype_final'])
        )
        
        # Load cell ids.
        cell_ids = list(np.load(Path(data_path, f'{mode}_ids.npy')))

        feature_files = [
            Path(data_path, 'skeletons', str(cell_id), 'features.npy')
            for cell_id in cell_ids
        ]
        sample_file = next((path for path in feature_files if path.exists()), None)
        if sample_file is None:
            raise FileNotFoundError(f'No feature files found for {mode} split')

        self.feature_dim = config['data']['feat_dim']
        if self.feature_dim not in (3, 4):
            raise ValueError(f"data.feat_dim must be 3 or 4, got {self.feature_dim}")
        available_features = np.load(sample_file, mmap_mode='r').shape[1]
        if available_features < self.feature_dim:
            raise ValueError(
                f'Config asks for {self.feature_dim} features, but {sample_file} '
                f'has only {available_features}'
            )

        self.radius_min = None
        self.radius_max = None
        if self.feature_dim == 4:
            train_ids = np.load(Path(data_path, 'train_ids.npy')).astype(str)
            train_files = [
                Path(data_path, 'skeletons', str(cell_id), 'features.npy')
                for cell_id in train_ids
            ]
            self.radius_min, self.radius_max = np.inf, -np.inf
            for path in train_files:
                if path.exists():
                    radii = np.load(path, mmap_mode='r')[:, 3]
                    if not np.isfinite(radii).all():
                        raise ValueError(f'Non-finite r_median values in {path}')
                    self.radius_min = min(self.radius_min, radii.min())
                    self.radius_max = max(self.radius_max, radii.max())
            if not np.isfinite(self.radius_min):
                raise ValueError('No training r_median values found')

        # Load graphs.
        self.manager = Manager()
        self.cells = self.manager.dict()
        count = 0
        for cell_id in tqdm(cell_ids):
            # Adapt for datasets where this is not true.
            soma_id = 0

            features = np.load(Path(data_path, 'skeletons', str(cell_id), 'features.npy'))
            if features.ndim != 2 or features.shape[1] < self.feature_dim:
                raise ValueError(
                    f'Expected at least {self.feature_dim} features for cell {cell_id}, '
                    f'got shape {features.shape}'
                )
            features = features[:, :self.feature_dim]
            if self.feature_dim == 4:
                span = self.radius_max - self.radius_min
                features[:, 3] = 0 if span == 0 else (features[:, 3] - self.radius_min) / span
            original_n_nodes = len(features)
            with open(Path(data_path, 'skeletons', str(cell_id), 'neighbors.pkl'), 'rb') as f:
                neighbors = pickle.load(f)

            assert len(features) == len(neighbors)

            if len(features) >= self.n_nodes or self.inference:
                
                # Subsample graphs for faster processing during training.
                neighbors, not_deleted = subsample_graph(neighbors=neighbors, 
                                                         not_deleted=set(range(len(neighbors))), 
                                                         keep_nodes=config['data']['cache_nodes'], 
                                                         protected=[soma_id])
                # Remap neighbor indices to 0..999.
                neighbors, subsampled2new = remap_neighbors(neighbors)
                soma_id = subsampled2new[soma_id]

                # Keep all node features; positional encodings use the graph.
                features = features[
                    list(subsampled2new.keys()), :self.feature_dim
                ]

                leaf_branch_nodes = get_leaf_branch_nodes(neighbors)
                # Using the distances we can infer the direction of an edge.
                distances = compute_node_distances(soma_id, neighbors)

                item = {
                    'cell_id': cell_id,
                    'subtype': subtype_by_cell_id[cell_id],
                    'features': features, 
                    'neighbors': neighbors,
                    'distances': distances,
                    'soma_id': soma_id,
                    'leaf_branch_nodes': leaf_branch_nodes,
                    'original_n_nodes': original_n_nodes
                }

                self.cells[count] = item
                count += 1

        # normalize cell counts
        original_counts = [
            item['original_n_nodes']
            for item in self.cells.values()
        ]

        self.log_n_min = np.log1p(min(original_counts))
        self.log_n_max = np.log1p(max(original_counts))

        self.num_samples = len(self.cells)
        self.sampling_labels = [
            self.cells[i]['subtype']
            for i in range(self.num_samples)
        ]

    def _normalize_node_count(self, n):
        log_n = np.log1p(n)

        return (log_n - self.log_n_min) / (
            self.log_n_max - self.log_n_min + 1e-8
        )

    def __len__(self):
        return self.num_samples

    def _delete_subbranch(self, neighbors, soma_id, distances, leaf_branch_nodes):

        leaf_branch_nodes = set(leaf_branch_nodes)
        not_deleted = set(range(len(neighbors))) 
        for i in range(self.n_drop_branch):
            neighbors, drop_nodes = drop_random_branch(leaf_branch_nodes, neighbors, distances, keep_nodes=self.n_nodes)
            not_deleted -= drop_nodes
            leaf_branch_nodes -= drop_nodes
            
            if len(leaf_branch_nodes) == 0:
                break

        return not_deleted

    def _reduce_nodes(self, neighbors, soma_id, distances, leaf_branch_nodes):
        neighbors2 = {k: set(v) for k, v in neighbors.items()}

        # Delete random branches.
        not_deleted = self._delete_subbranch(neighbors2, soma_id, distances, leaf_branch_nodes)

        # Subsample graphs to fixed number of nodes.
        protected = soma_id if isinstance(soma_id, (list, tuple, set)) else [soma_id]
        neighbors2, not_deleted = subsample_graph(neighbors=neighbors2, not_deleted=not_deleted, keep_nodes=self.n_nodes, protected=protected)

        # Compute new adjacency matrix.
        adj_matrix = neighbors_to_adjacency_torch(neighbors2, not_deleted)
        assert adj_matrix.shape == (self.n_nodes, self.n_nodes), '{} {}'.format(adj_matrix.shape)
        
        return neighbors2, adj_matrix, not_deleted
    
    
    def _augment_node_position(self, features):
        # Extract positional features (xyz-position).
        pos = features[:, :3]

        # Rotate (random 3D rotation or rotation around specific axis).
        rot_pos = rotate_graph(pos, axis=self.rotation_axis)

        # Randomly jitter node position.
        jittered_pos = jitter_node_pos(rot_pos, scale=self.jitter_var)
        
        # Translate neuron position as a whole.
        jittered_pos = translate_soma_pos(jittered_pos, scale=self.translate_var)

        # Randomly scale the XY hull slightly inward or outward.
        # jittered_pos = scale_hull(jittered_pos, scale_range=self.hull_scale_range)
        
        features[:, :3] = jittered_pos

        return features
    

    def _augment(self, cell):

        features = cell['features']
        neighbors = cell['neighbors']
        distances = cell['distances']

        # Reduce nodes to N == n_nodes via subgraph deletion and subsampling.
        neighbors2, adj_matrix, not_deleted = self._reduce_nodes(neighbors, [int(cell['soma_id'])], distances, cell['leaf_branch_nodes'])

        # Extract features of remaining nodes.
        new_features = features[not_deleted].copy()
       
        # Augment node position via rotation and jittering.
        new_features = self._augment_node_position(new_features)

        return new_features, adj_matrix
    
    def __getsingleitem__(self, index): 
        cell = self.cells[index]

        features = cell['features']
        neighbors = cell['neighbors']
        distances = cell['distances']
        neighbors2, adj_matrix, not_deleted = self._reduce_nodes(neighbors, [int(cell['soma_id'])], distances, cell['leaf_branch_nodes'])
        new_features = features[not_deleted].copy()

        return new_features, adj_matrix, cell['cell_id']
    
    
    def __getitem__(self, index): 
        cell = self.cells[index]

        # Compute two different views through augmentations.
        features1, adj_matrix1 = self._augment(cell)
        features2, adj_matrix2 = self._augment(cell)

        return features1, features2, adj_matrix1, adj_matrix2

    def get_cell_by_id(self, cell_id):
        for idx, cell in self.cells.items():
            if str(cell['cell_id']) == str(cell_id):
                return cell

        raise KeyError(f"Cell ID {cell_id} not found")

    def get_augmented_cell_by_id(self, cell_id):
        for idx, cell in self.cells.items():
            if str(cell['cell_id']) == str(cell_id):
                features, adj_matrix = self._augment(cell)
                cell['features'] = features
                cell['neighbors'] = adjacency_to_neighbors(adj_matrix.numpy())
                cell['adj_matrix'] = adj_matrix
                return cell

        raise KeyError(f"Cell ID {cell_id} not found")
    

def build_dataloader(config, use_cuda=torch.cuda.is_available()):

    kwargs = {'num_workers':config['data']['num_workers'], 'pin_memory':True, 'persistent_workers': True} if use_cuda else {}

    train_dataset = GraphDataset(config, mode='train')
    labels = train_dataset.sampling_labels
    unique_labels, counts = np.unique(labels, return_counts=True)
    # class_counts = dict(zip(unique_labels, counts))

    # alpha = config['data']['oversampling_alpha']

    # sample_weights = [
    #     1.0 / (class_counts[label] ** alpha)
    #     for label in labels
    # ]

    sampler = BalancedSampler(labels)

    # sampler = WeightedRandomSampler(
    #     weights=sample_weights,
    #     num_samples=len(train_dataset),
    #     replacement=True
    # )

    train_loader = DataLoader(
            train_dataset,
            batch_size=config['data']['batch_size'], 
            shuffle=True,
            # sampler=sampler,
            # drop_last=False,
            drop_last=True,
            **kwargs)

    val_dataset = GraphDataset(config, mode='val')
    batch_size = val_dataset.num_samples if val_dataset.__len__() < config['data']['batch_size'] else config['data']['batch_size']
    val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=True,
            **kwargs)

    return train_loader, val_loader
