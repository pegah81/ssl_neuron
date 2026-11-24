import torch
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
from collections import defaultdict
from scipy.spatial.transform import Rotation as R
from sklearn.metrics.pairwise import cosine_distances
from collections import Counter
from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import Sampler

import os
import pickle
import io
from contextlib import redirect_stdout
from functools import wraps
import pandas as pd
from pathlib import Path
from tqdm import tqdm


class AverageMeter(object):
    """ Computes and stores the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
        
def subsample_graph(neighbors=None, not_deleted=None, keep_nodes=200, protected=[0]):
    """ 
    Subsample graph.

    Args:
        neighbors: dict of neighbors per node
        not_deleted: list of nodes, who did not get deleted in previous processing steps
        keep_nodes: number of nodes to keep in graph
        protected: nodes to be excluded from subsampling
    """
    if neighbors is not None:
        k_nodes = len(neighbors)
    else:
        raise ValueError('neighbors must be provided')

    # protect soma node from being removed
    protected = set(protected)

    # indices as set in random order
    perm = torch.randperm(k_nodes).tolist()
    all_indices = np.array(list(not_deleted))[perm].tolist()
    deleted = set()
    
    while len(deleted) < k_nodes - keep_nodes:

        while True:
            if len(all_indices) == 0:
                assert len(not_deleted) > keep_nodes, len(not_deleted)
                remaining = list(not_deleted - deleted)
                perm = torch.randperm(len(remaining)).tolist()
                all_indices = np.array(remaining)[perm].tolist()

            idx = all_indices.pop()

            if idx not in deleted and len(neighbors[idx]) < 3 and idx not in protected:
                break

        if len(neighbors[idx]) == 2:
            n1, n2 = neighbors[idx]
            neighbors[n1].remove(idx)
            neighbors[n2].remove(idx)
            neighbors[n1].add(n2)
            neighbors[n2].add(n1)
        elif len(neighbors[idx]) == 1:
            n1 = neighbors[idx].pop()
            neighbors[n1].remove(idx)

        del neighbors[idx]
        deleted.add(idx)

    not_deleted = list(not_deleted - deleted)
    return neighbors, not_deleted



# def rotate_graph(pos_matrix, axis=None):
#     ''' Randomly rotate graph in xyz-direction.

#     Args:
#         pos_matrix: Matrix with xyz-node positions (N x 3).
#         axis: Axis around which to rotate. Defaults to `None`,
#             in which case no rotation is performed.
#     '''
#     if axis is None:
#         return pos_matrix

#     rotation_matrix = R.random().as_matrix()
    
#     if axis == 'x':
#         rotation_matrix[0, 1] = 0
#         rotation_matrix[0, 2] = 0
#         rotation_matrix[0, 0] = 1
#         rotation_matrix[1, 0] = 0
#         rotation_matrix[2, 0] = 0
#     elif axis == 'y':
#         rotation_matrix[0, 1] = 0
#         rotation_matrix[1, 0] = 0
#         rotation_matrix[1, 1] = 1
#         rotation_matrix[1, 2] = 0
#         rotation_matrix[2, 1] = 0
#     elif axis == 'z':
#         rotation_matrix[0, 2] = 0
#         rotation_matrix[1, 2] = 0
#         rotation_matrix[2, 2] = 1
#         rotation_matrix[2, 0] = 0
#         rotation_matrix[2, 1] = 0

#     rot_pos_matrix = pos_matrix @ rotation_matrix
#     return rot_pos_matrix

def rotate_graph(pos_matrix, axis=None):
    """Randomly rotate graph around one configured axis without distortion."""
    if axis is None:
        return pos_matrix

    angle = np.random.uniform(0, 2 * np.pi)
    c, s = np.cos(angle), np.sin(angle)

    if axis == "x":
        rotation_matrix = np.array([
            [1,  0,  0],
            [0,  c, -s],
            [0,  s,  c],
        ])
    elif axis == "y":
        rotation_matrix = np.array([
            [ c,  0,  s],
            [ 0,  1,  0],
            [-s,  0,  c],
        ])
    elif axis == "z":
        rotation_matrix = np.array([
            [c, -s,  0],
            [s,  c,  0],
            [0,  0,  1],
        ])

    return pos_matrix @ rotation_matrix


def jitter_node_pos(node_positions, scale=(1,1,0.2)):
    """ 
    Randomly jitter nodes in xyz-direction.

    Args:
        node_positions: Matrix with xyz-node positions (N x 3).
        scale: Scale factor for jittering.
    """
    noise = np.random.randn(*node_positions.shape) * scale
    return node_positions + noise


def translate_soma_pos(node_positions, scale=(5, 5, 2)):
    """
    Randomly translate the position of the entire graph.
    """
    new_node_features = node_positions.copy()
    jitter = torch.randn(3).numpy() * scale
    new_node_features[:, :3] += jitter
    return new_node_features


def scale_hull(node_positions, scale_range=(-0.05, 0.05)):
    """Randomly scale the XY hull around the cell center.

    ``scale_range`` is a fractional change: negative values shrink the hull,
    zero leaves it unchanged, and positive values expand it. Z is unchanged.
    """
    if len(node_positions) == 0:
        return node_positions

    expanded = node_positions.copy()
    center = expanded[:, :2].mean(axis=0, keepdims=True)
    scale = 1.0 + np.random.uniform(*scale_range)
    expanded[:, :2] = center + (expanded[:, :2] - center) * scale
    return expanded


def neighbors_to_adjacency_torch(neighbors, not_deleted):
    """ Create adjacency matrix from list of non-empty neighbors.
    """
    node_map = {n: i for i, n in enumerate(not_deleted)}

    n_nodes = len(not_deleted)

    new_adj_matrix = torch.zeros((n_nodes, n_nodes), dtype=float)
    for ii in neighbors.keys():
        for jj in neighbors[ii]:
            i, j = node_map[ii], node_map[jj]
            new_adj_matrix[i, i] = True  # diagonal if needed
            new_adj_matrix[i, j] = True
            new_adj_matrix[j, i] = True

    return new_adj_matrix


def neighbors_to_adjacency(neighbors, not_deleted):
    """ 
    Create adjacency matrix from list of non-empty neighbors. 

    Args:
        neighbors: Dict of neighbors per node.
        not_deleted: List of nodes, who did not get deleted in previous processing steps.
    """
    node_map = {n: i for i, n in enumerate(not_deleted)}
    
    n_nodes = len(not_deleted)
    
    new_adj_matrix = np.zeros((n_nodes, n_nodes))
    for ii in neighbors.keys():
        for jj in neighbors[ii]:
            i, j = node_map[ii], node_map[jj]
            new_adj_matrix[i, i] = True  # diagonal if needed
            new_adj_matrix[i, j] = True
            new_adj_matrix[j, i] = True   
            
    return new_adj_matrix


def adjacency_to_neighbors(adj_matrix):
    """ 
    Create list of non-empty neighbors from adjacancy matrix. 
    Args:
        adj_matrix: adjacency matrix (N x N)
    """
    # Remove diagonal to avoid self-neighbors.
    a, b = np.where(adj_matrix - np.eye(len(adj_matrix)) == 1)
    neigh = dict()
    for _a, _b in zip(a,b):
        if _a not in neigh:
            neigh[_a] = set()
        neigh[_a].add(_b)
    return neigh


def compute_eig_lapl_torch_batch(adj_matrix, pos_enc_dim=32):
    """ Compute positional encoding using graph laplacian.
        Adapted from https://github.com/graphdeeplearning/benchmarking-gnns/blob/ef8bd8c7d2c87948bc1bdd44099a52036e715cd0/data/molecules.py#L147-L168.
    
    Args:
        adj_matrix: Adjacency matrix (B x N x N).
        pos_enc_dim: Output dimensions of positional encoding.
    """
    b, n, _ = adj_matrix.size()
    
    # Laplacian
    A = adj_matrix.float()
    degree_matrix = A.sum(axis=1).clip(1)
    N = torch.diag_embed(degree_matrix ** -0.5)
    L = torch.eye(n, device=A.device)[None, ].repeat(b, 1, 1) - (N @ A) @ N

    # Eigenvectors
    eig_val, eig_vec = torch.linalg.eigh(L)
    eig_vec = torch.flip(eig_vec, dims=[2])
    pos_enc = eig_vec[:, :, 1:pos_enc_dim + 1]

    if pos_enc.size(2) < pos_enc_dim:
        pos_enc = torch.cat([pos_enc, torch.zeros(pos_enc.size(0), pos_enc.size(1), pos_enc_dim - pos_enc.size(2), device=A.device)], dim=2)

    return pos_enc


def get_leaf_branch_nodes(neighbors):
    """"
    Create list of candidates for leaf and branching nodes.
    Args:
        neighbors: dict of neighbors per node
    """
    all_nodes = list(neighbors.keys())
    leafs = [i for i in all_nodes if len(neighbors[i]) == 1]
    
    candidates = leafs
    next_nodes = []
    for l in leafs:
        next_nodes += [n for n in neighbors[l] if len(neighbors[n]) == 2]

    while next_nodes:
        s = next_nodes.pop(0) 
        candidates.append(s)
        next_nodes += [n for n in neighbors[s] if len(neighbors[n]) == 2 and n not in candidates and n not in next_nodes] 
        
    return candidates


def compute_node_distances(idx, neighbors):
    """"
    Computation of node degree.
    Args:
        idx: index of node
        neighbors: dict of neighbors per node
    """
    queue = []
    queue.append(idx)

    degree = dict()
    degree[idx] = 0

    while queue:
        s = queue.pop(0) 
        prev_dist = degree[s]

        for neighbor in neighbors[s]:
              if neighbor not in degree:
                queue.append(neighbor)
                degree[neighbor] = prev_dist + 1

    return degree


def drop_random_branch(nodes, neighbors, distances, keep_nodes=200):
    """ 
    Removes a terminal branch. Starting nodes should be between
    branching node and leaf (see leaf_branch_nodes)

    Args:
        nodes: List of nodes of the graph
        neighbors: Dict of neighbors per node
        distances: Dict of distances of nodes to origin
        keep_nodes: Number of nodes to keep in graph
    """
    start = list(nodes)[torch.randint(len(nodes), (1,)).item()]
    to = list(neighbors[start])[0]

    if distances[start] > distances[to]:
        start, to = to, start

    drop_nodes = [to]
    next_nodes = [n for n in neighbors[to] if n != start]

    while next_nodes:
        s = next_nodes.pop(0) 
        drop_nodes.append(s)
        next_nodes += [n for n in neighbors[s] if n not in drop_nodes] 

    if len(neighbors) - len(drop_nodes) < keep_nodes:
        return neighbors, set()
    else:
        # Delete nodes.
        for key in drop_nodes:
            if key in neighbors:
                for k in neighbors[key]:
                    neighbors[k].remove(key)
                del neighbors[key]

        return neighbors, set(drop_nodes)
    
    
def traverse_dir(start, to, neighbors):
    """ 
    Traverse branch start at node 'start' in direction of node 'to'. 
    Args:
        start: start node
        to: destination node
        neighbors: dict of neighbors per node
    """
    visited = [start, to]
    next_nodes = [n for n in neighbors[to] if n != start]

    while next_nodes:
        s = next_nodes.pop(0) 
        visited.append(s)
        next_nodes += [n for n in neighbors[s] if n not in visited]
    
    return visited


def remap_neighbors(x):
    """ 
    Remap node indices to be between 0 and the number of nodes.

    Args:
        x: Dict of node id mapping to the node's neighbors.
    Returns:
        ordered_x: Dict with neighbors with new node ids.
        subsampled2new: Mapping between old and new indices (dict).
    """
    # Create maps between new and old indices.
    subsampled2new = {k: i for i, k in enumerate(sorted(x))}

    # Re-map indices to 1..N.
    ordered_x = {i: x[k] for i, k in enumerate(sorted(x))}

    # Re-map keys of neighbors
    for k in ordered_x:
        ordered_x[k] = {subsampled2new[x] for x in ordered_x[k]}

    return ordered_x, subsampled2new



def plot_neuron(neighbors, node_feats, ax1=0, ax2=1, soma_id=0, ax=None):
    """ Plot graph of 3D neuronal morphology. """   
    colors = list(sns.dark_palette('#69d', n_colors=4))
    _, dim = node_feats.shape
    
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    
    ax.set_aspect('equal')

    for i, neigh in neighbors.items():
        for j in neigh:
            n1, n2 = node_feats[i], node_feats[j]
            # xyz + r_median has four columns; compartment one-hot features,
            # when present, begin at column four.
            c = colors[np.argmax(n2[4:8])] if dim >= 8 else colors[1]
            ax.plot([n1[ax1], n2[ax1]], [n1[ax2], n2[ax2]], color=c, linewidth=1)

    if soma_id is not None:
        ax.scatter(node_feats[soma_id][ax1], node_feats[soma_id][ax2], color=colors[0], s=10, zorder=10)
    
    sns.despine(ax=ax, trim=1)


def compare_augment_plot(cell_id, dset, merged_df, figsize=(12, 10)):
    """Compare one raw skeleton with its augmented view in XY and XZ."""
    cell_dir = Path(dset.config['data']['path']) / 'skeletons' / str(cell_id)
    raw_features = np.load(cell_dir / 'features.npy')[:, :3]
    with open(cell_dir / 'neighbors.pkl', 'rb') as f:
        raw_neighbors = pickle.load(f)
    augmented_cell = dset.get_augmented_cell_by_id(cell_id)

    labels = merged_df.loc[
        merged_df['cell'].astype(str) == str(cell_id), 'celltype_final'
    ].dropna()
    celltype = labels.iloc[0] if len(labels) else 'Unknown subtype'

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    augmented_features = augmented_cell['features'][:, :3]
    all_features = np.concatenate([raw_features, augmented_features])

    for row, (x_axis, y_axis, view) in enumerate([(0, 1, 'XY'), (0, 2, 'XZ')]):
        plot_neuron(
            raw_neighbors, raw_features, ax1=x_axis, ax2=y_axis,
            soma_id=0, ax=axes[row, 0]
        )
        axes[row, 0].set_title(f'Raw skeleton ({view})')
        plot_neuron(
            augmented_cell['neighbors'], augmented_cell['features'],
            ax1=x_axis, ax2=y_axis, soma_id=None, ax=axes[row, 1]
        )
        axes[row, 1].set_title(f'Augmented view ({view})')

        x_min, x_max = all_features[:, x_axis].min(), all_features[:, x_axis].max()
        y_min, y_max = all_features[:, y_axis].min(), all_features[:, y_axis].max()
        x_pad = (x_max - x_min) * 0.05 or 1
        y_pad = (y_max - y_min) * 0.05 or 1
        for ax in axes[row]:
            ax.set_xlim(x_min - x_pad, x_max + x_pad)
            ax.set_ylim(y_min - y_pad, y_max + y_pad)
            ax.tick_params(labelbottom=True, labelleft=True)
            ax.set_xlabel('X' if x_axis == 0 else 'Y')
            ax.set_ylabel('Y' if y_axis == 1 else 'Z')

    fig.suptitle(f'Cell ID: {cell_id} | Cell type: {celltype}')
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig, axes


def _silence_plot_stdout(func):
    """Suppress incidental console output while building a plot."""
    @wraps(func)
    def wrapped(*args, **kwargs):
        with redirect_stdout(io.StringIO()):
            return func(*args, **kwargs)
    return wrapped


@_silence_plot_stdout
def plot_augmentation_effects(
    cell_ids, config, merged_df, augmentations=None, figsize=None
):
    """Plot each cell's full skeleton and isolated configured augmentations.

    ``cell_ids`` may be one cell ID or a sequence of IDs. Each cell has a
    shared cell ID and subtype heading, followed by XY/XZ row pairs with at
    most four variants per row. ``augmentations`` selects which variants to
    include: ``downsample``, ``n_nodes``, ``rotation``, ``jitter``,
    ``translation``, and ``branch_drop``. Rotation is always around Z. The
    full skeleton is always included.
    """
    if isinstance(cell_ids, (str, bytes)) or np.isscalar(cell_ids):
        cell_ids = [cell_ids]
    else:
        cell_ids = list(cell_ids)
    if not cell_ids:
        raise ValueError("cell_ids must contain at least one cell ID")

    available_augmentations = {
        'downsample': 'Downsampled to cache_nodes',
        'n_nodes': 'Downsampled to n_nodes',
        'rotation': 'Rotation around Z only',
        'jitter': 'Jitter only',
        'translation': 'Translation only',
        'branch_drop': 'Branch drop only',
    }
    aliases = {
        'downsampling': 'downsample',
        'cache_nodes': 'downsample',
        'model_size': 'n_nodes',
        'rotate': 'rotation',
        'translate': 'translation',
        'drop_branch': 'branch_drop',
        'branch': 'branch_drop',
    }
    if augmentations is None:
        selected_augmentations = list(available_augmentations)
    elif isinstance(augmentations, str):
        selected_augmentations = [augmentations]
    else:
        selected_augmentations = list(augmentations)
    selected_augmentations = [
        aliases.get(name, name) for name in selected_augmentations
    ]
    unknown_augmentations = set(selected_augmentations).difference(
        available_augmentations
    )
    if unknown_augmentations:
        raise ValueError(
            f"Unknown augmentations: {sorted(unknown_augmentations)}. "
            f"Choose from {list(available_augmentations)}."
        )
    if len(set(selected_augmentations)) != len(selected_augmentations):
        raise ValueError("augmentations must not contain duplicates")

    required_columns = {'celltype_final'}
    id_column = 'cell' if 'cell' in merged_df.columns else 'cell_id'
    required_columns.add(id_column)
    missing_columns = required_columns.difference(merged_df.columns)
    if missing_columns:
        raise ValueError(
            f"merged_df is missing required columns: {sorted(missing_columns)}"
        )

    # Accept either the raw config dictionary or a GraphDataset instance.
    config_dict = config if isinstance(config, dict) else config.config
    data_config = config_dict['data']
    cache_nodes = int(data_config['cache_nodes'])
    n_nodes = int(data_config['n_nodes'])
    n_drop_branch = int(data_config['n_drop_branch'])
    if cache_nodes <= 0 or n_nodes <= 0 or n_drop_branch < 0:
        raise ValueError(
            "cache_nodes and n_nodes must be positive, and n_drop_branch "
            "must be non-negative"
        )

    def remap_graph(neighbors, features):
        old_node_ids = sorted(neighbors)
        remapped_neighbors, _ = remap_neighbors(neighbors)
        remapped_features = np.asarray(
            [features[node_id] for node_id in old_node_ids], dtype=float
        )
        return remapped_features, remapped_neighbors

    def reduce_to_model_size(neighbors):
        reduced_neighbors, _ = subsample_graph(
            neighbors={node: set(adjacent) for node, adjacent in neighbors.items()},
            not_deleted=set(neighbors),
            keep_nodes=n_nodes,
            protected=[0],
        )
        return reduced_neighbors

    data_path = Path(data_config['path'])
    variant_titles = {'full': 'Full skeleton'}
    variant_titles.update(available_augmentations)
    variant_titles['downsample'] = f'Downsampled to cache_nodes={cache_nodes}'
    variant_titles['n_nodes'] = f'Downsampled to n_nodes={n_nodes}'
    variant_titles['branch_drop'] = (
        f'Branch drop only (n={n_drop_branch})'
    )
    variant_keys = ['full'] + selected_augmentations
    max_panels_per_row = 4
    ncols = min(max_panels_per_row, len(variant_keys))
    groups_per_cell = int(np.ceil(len(variant_keys) / max_panels_per_row))
    rows_per_cell = 1 + 2 * groups_per_cell
    if figsize is None:
        figsize = (
            4.8 * ncols,
            (0.55 + 9.0 * groups_per_cell) * len(cell_ids),
        )
    fig = plt.figure(figsize=figsize)
    cell_grid = fig.add_gridspec(
        rows_per_cell * len(cell_ids),
        ncols,
        height_ratios=[
            ratio
            for _ in cell_ids
            for ratio in ([0.55] + [4.5, 4.5] * groups_per_cell)
        ],
        hspace=0.34,
        wspace=0.18,
    )
    axes = np.full(
        (2 * groups_per_cell * len(cell_ids), ncols), None, dtype=object
    )

    for row, cell_id in enumerate(cell_ids):
        cell_dir = data_path / 'skeletons' / str(cell_id)
        raw_features = np.asarray(np.load(cell_dir / 'features.npy'))[:, :3]
        with open(cell_dir / 'neighbors.pkl', 'rb') as f:
            raw_neighbors = pickle.load(f)
        raw_neighbors = {
            int(node): set(map(int, adjacent))
            for node, adjacent in raw_neighbors.items()
        }
        if len(raw_features) != len(raw_neighbors):
            raise ValueError(
                f"Cell {cell_id} has {len(raw_features)} feature rows but "
                f"{len(raw_neighbors)} graph nodes"
            )

        cached_neighbors, _ = subsample_graph(
            neighbors={
                node: set(adjacent)
                for node, adjacent in raw_neighbors.items()
            },
            not_deleted=set(raw_neighbors),
            keep_nodes=cache_nodes,
            protected=[0],
        )
        cached_features, cached_neighbors = remap_graph(
            cached_neighbors, raw_features
        )

        # Build the common model-sized graph once so jitter and translation
        # are compared on the same sampled nodes and topology.
        model_neighbors = reduce_to_model_size(cached_neighbors)
        model_features, model_neighbors = remap_graph(
            model_neighbors, cached_features
        )

        jitter_features = jitter_node_pos(
            model_features.copy(), scale=data_config['jitter_var']
        )
        rotated_features = rotate_graph(
            model_features.copy(), axis='z'
        )
        translated_features = translate_soma_pos(
            model_features.copy(), scale=data_config['translate_var']
        )

        branch_neighbors = {
            node: set(adjacent) for node, adjacent in cached_neighbors.items()
        }
        not_deleted = set(branch_neighbors)
        leaf_branch_nodes = set(get_leaf_branch_nodes(branch_neighbors))
        distances = compute_node_distances(0, branch_neighbors)
        for _ in range(n_drop_branch):
            if not leaf_branch_nodes:
                break
            branch_neighbors, dropped_nodes = drop_random_branch(
                leaf_branch_nodes,
                branch_neighbors,
                distances,
                keep_nodes=n_nodes,
            )
            not_deleted -= dropped_nodes
            leaf_branch_nodes -= dropped_nodes
        branch_neighbors, _ = subsample_graph(
            neighbors=branch_neighbors,
            not_deleted=not_deleted,
            keep_nodes=n_nodes,
            protected=[0],
        )
        branch_features, branch_neighbors = remap_graph(
            branch_neighbors, cached_features
        )

        labels = merged_df.loc[
            merged_df[id_column].astype(str) == str(cell_id),
            'celltype_final',
        ].dropna()
        celltype = labels.iloc[0] if len(labels) else 'Unknown subtype'
        cell_start = rows_per_cell * row
        title_ax = fig.add_subplot(cell_grid[cell_start, :])
        title_ax.set_axis_off()
        title_ax.text(
            0.5,
            0.5,
            f'Cell {cell_id} | {celltype}',
            ha='center',
            va='center',
            fontsize=16,
            fontweight='bold',
        )
        variant_data = {
            'full': (raw_features, raw_neighbors),
            'downsample': (cached_features, cached_neighbors),
            'n_nodes': (model_features, model_neighbors),
            'rotation': (rotated_features, model_neighbors),
            'jitter': (jitter_features, model_neighbors),
            'translation': (translated_features, model_neighbors),
            'branch_drop': (branch_features, branch_neighbors),
        }

        for variant_index, variant_key in enumerate(variant_keys):
            variant_group, variant_col = divmod(
                variant_index, max_panels_per_row
            )
            variant_title = variant_titles[variant_key]
            features, neighbors = variant_data[variant_key]
            for view_index, (x_axis, y_axis, view) in enumerate(
                [(0, 1, 'XY'), (0, 2, 'XZ')]
            ):
                axes_row = (
                    row * 2 * groups_per_cell
                    + variant_group * 2
                    + view_index
                )
                grid_row = cell_start + 1 + variant_group * 2 + view_index
                axes[axes_row, variant_col] = fig.add_subplot(
                    cell_grid[grid_row, variant_col]
                )
                ax = axes[
                    axes_row,
                    variant_col,
                ]
                plot_neuron(
                    neighbors,
                    features,
                    ax1=x_axis,
                    ax2=y_axis,
                    soma_id=0,
                    ax=ax,
                )
                ax.set_title(
                    f'{variant_title} | {view}',
                    fontsize=10,
                )
                ax.set_xlabel('X')
                ax.set_ylabel('Y' if y_axis == 1 else 'Z')

    fig.subplots_adjust(left=0.04, right=0.99, top=0.99, bottom=0.04)
    return fig, axes


def plot_neurons(dset, cell_ids, merged_df=None, ax1=0, ax2=1, soma_id=0, augment=False):
    """Plot cells from a GraphDataset, with up to four plots per row.

    Graph data is read from ``dset.get_cell_by_id``. If ``merged_df`` is
    provided, subtype labels are looked up using its ``cell`` and
    ``celltype_final`` columns.
    """
    ax1, ax2 = int(ax1), int(ax2)
    n_cells = len(cell_ids)
    if n_cells == 0:
        raise ValueError("cell_ids must contain at least one cell ID")
    ncols = min(n_cells, 4)
    nrows = int(np.ceil(n_cells / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(5 * ncols, 5 * nrows),
        squeeze=False
    )
    colors = list(sns.dark_palette('#69d', n_colors=4))
    celltype_by_id = {}
    if merged_df is not None:
        required_columns = {'cell', 'celltype_final'}
        missing_columns = required_columns.difference(merged_df.columns)
        if missing_columns:
            raise ValueError(
                f"merged_df is missing required columns: {sorted(missing_columns)}"
            )
        celltype_by_id = dict(zip(
            merged_df['cell'].astype(str), merged_df['celltype_final']
        ))

    for idx, cell_id in enumerate(cell_ids):
        if augment:
            cell = dset.get_augmented_cell_by_id(cell_id)
        else:
            cell = dset.get_cell_by_id(cell_id)
        features = cell['features']
        neighbors = cell['neighbors']
        soma = cell['soma_id']
        ax = axes.flat[idx]
        _, dim = features.shape
        ax.set_aspect('equal')
        for i, neigh in neighbors.items():
            for j in neigh:
                n1, n2 = features[i], features[j]

                c = colors[np.argmax(n2[4:8])] if dim >= 8 else colors[1]

                ax.plot(
                    [n1[ax1], n2[ax1]],
                    [n1[ax2], n2[ax2]],
                    color=c,
                    linewidth=1
                )
        ax.scatter(
            features[soma][ax1],
            features[soma][ax2],
            color=colors[0],
            s=15,
            zorder=10
        )
        celltype = celltype_by_id.get(str(cell_id), cell.get('subtype', cell.get('celltype_final')))
        title = f'Cell ID: {cell_id}'
        if celltype is not None and not pd.isna(celltype):
            title += f'\n{celltype}'
        ax.set_title(title)
        sns.despine(ax=ax, trim=True)

    for idx in range(n_cells, nrows * ncols):
        axes.flat[idx].set_visible(False)

    plt.tight_layout()
    return fig, axes


def plot_random_neurons_by_type(
    cell_meta,
    dset,
    cell_types,
    n_samples=5,
    random_state=42
):
    rng = np.random.default_rng(random_state)

    cell_id_to_idx = {
        str(cell['cell_id']): idx
        for idx, cell in dset.cells.items()
    }

    for cell_type in cell_types:

        subset = cell_meta[
            cell_meta['celltype_final'] == cell_type
        ]

        available_ids = [
            str(cell_id)
            for cell_id in subset['cell_id']
            if str(cell_id) in cell_id_to_idx
        ]

        if not available_ids:
            print(f"No cells available for {cell_type}")
            continue

        sampled_ids = rng.choice(
            available_ids,
            size=min(n_samples, len(available_ids)),
            replace=False
        )

        n = len(sampled_ids)

        fig, axes = plt.subplots(
            2, n,
            figsize=(4 * n, 8),
            squeeze=False
        )

        for j, cell_id in enumerate(sampled_ids):

            idx = cell_id_to_idx[cell_id]
            feat, neigh, _ = dset.__getsingleitem__(idx)

            plot_neuron(
                neighbors=neigh,
                node_feats=feat,
                ax1=0,
                ax2=1,
                soma_id=0,
                ax=axes[0, j]
            )

            axes[0, j].set_title(f"{cell_id}\nXY view", fontsize=9)

            plot_neuron(
                neighbors=neigh,
                node_feats=feat,
                ax1=0,
                ax2=2,
                soma_id=0,
                ax=axes[1, j]
            )

            axes[1, j].set_title(f"{cell_id}\nXZ view", fontsize=9)

        fig.suptitle(
            f"{cell_type} — {n} randomly selected cells",
            fontsize=14
        )

        plt.tight_layout()
        plt.show()

    
def plot_tsne(z, labels, targets, colors=None):
    """ Plot t-SNE clustering. """
    u_labels = np.unique(labels)
    fig = plt.figure(1, figsize=(8, 8))
    for label in u_labels:
        plt.scatter(z[labels == label, 0], 
                    z[labels==label, 1], 
                    s=20, 
                    label=str(targets[label]),
                    color=colors[label])
    plt.legend(bbox_to_anchor=(1,1))
    plt.axis('off')


def knn_accuracy(
    latents,
    subset=None,
    cell_types=None,
    k=5,
    balanced=True,
    print_val=False
):
    space = latents.dropna(subset=["latent", "celltype_final"]).copy()
    space = space[
        space["latent"].apply(lambda x: x is not None and len(x) > 0)
    ].copy()

    if subset is not None:
        df = subset.copy()
        df = df.dropna(subset=["latent", "celltype_final"])
        df = df[
            df["latent"].apply(lambda x: x is not None and len(x) > 0)
        ].copy()
    else:
        df = space.copy()

    if cell_types is not None:
        df = df[df["celltype_final"].isin(cell_types)].copy()

    if len(space) <= k:
        raise ValueError("Not enough cells in the whole space for the selected k.")

    if len(df) == 0:
        raise ValueError("No cells to evaluate.")

    X_space = np.stack(space["latent"].apply(np.asarray).to_numpy())
    X_eval = np.stack(df["latent"].apply(np.asarray).to_numpy())

    distances = cosine_distances(X_eval, X_space)

    space_indices = pd.Series(
        np.arange(len(space)), index=space["cell_id"]
    )
    for i, cell_id in enumerate(df["cell_id"]):
        if cell_id in space_indices.index:
            distances[i, space_indices[cell_id]] = np.inf

    nearest_indices = np.argsort(distances, axis=1)[:, :k]

    space_labels = space["celltype_final"].to_numpy()
    true_labels = df["celltype_final"].to_numpy()

    predictions = []
    for neighbors in nearest_indices:
        neighbor_labels = space_labels[neighbors]
        prediction = Counter(neighbor_labels).most_common(1)[0][0]
        predictions.append(prediction)

    if balanced:
        accuracy = balanced_accuracy_score(true_labels, predictions)
    else:
        accuracy = np.mean(true_labels == np.array(predictions))

    if print_val:
        print(f"Cells in whole space: {len(space)}")
        print(f"Cells evaluated: {len(df)}")
        print(f"Cell types: {cell_types if cell_types is not None else 'All'}")
        print(f"k: {k}")
        print(f"{'Balanced ' if balanced else ''}KNN accuracy: {accuracy:.4f} ({accuracy:.2%})")

    results = df[["cell_id", "celltype_final"]].copy()
    results["predicted_celltype"] = predictions
    results["correct"] = results["celltype_final"] == results["predicted_celltype"]

    return accuracy, results

def knn_predictions_by_celltype(space, evaluation, cell_types, k=5):
    predictions = []
    accuracies = []
    for cell_type in cell_types:
        evaluation_type = evaluation[evaluation["celltype_final"] == cell_type]
        if evaluation_type.empty:
            continue
        type_accuracy, type_results = knn_accuracy(
            space, subset=evaluation_type, cell_types=[cell_type],
            k=k, balanced=False, print_val=False
        )
        predictions.append(type_results)
        accuracies.append({"celltype": cell_type, "accuracy": type_accuracy})
    if not predictions:
        return pd.DataFrame(columns=["cell_id", "celltype_final", "predicted_celltype", "correct"]), pd.DataFrame(columns=["celltype", "accuracy"])
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(accuracies)

def save_trial_results(
        cell_meta,
        config,
        TRIAL,
        k=10
    ):

        accuracy_train, _ = knn_accuracy(
            cell_meta[(cell_meta["split"].isin(["train"]))],
            k=k,
            balanced=False
        )

        accuracy_val, _ = knn_accuracy(
            cell_meta[(cell_meta["split"].isin(["train", "val"]))],
            cell_meta[(cell_meta["split"].isin(["val"]))],
            k=k,
            balanced=False
        )

        accuracy_train_bal, _ = knn_accuracy(
            cell_meta[(cell_meta["split"].isin(["train"]))],
            k=k,
            balanced=True
        )

        accuracy_val_bal, _ = knn_accuracy(
            cell_meta[(cell_meta["split"].isin(["train"]))],
            cell_meta[(cell_meta["split"].isin(["val"]))],
            k=k,
            balanced=True
        )

        results = {
            "trial": TRIAL,
            "train_acc": accuracy_train,
            "val_acc": accuracy_val,
            "balanced_train_acc": accuracy_train_bal,
            "balanced_val_acc": accuracy_val_bal,
            "k": k,
            "confidence": None,
            "max_iter": config["optimizer"]["max_iter"],
            "lr": config["optimizer"]["lr"],
            "exp_decay": config["optimizer"]["exp_decay"],
            "n_nodes": config["data"]["n_nodes"],
            "jitter_var": str(config["data"]["jitter_var"]),
            "translate_var": str(config["data"]["translate_var"]),
            "rotation_axis": config["data"]["rotation_axis"],
            "n_drop_branch": config["data"]["n_drop_branch"],
            "dim": config["model"]["dim"],
            "depth": config["model"]["depth"],
            "n_head": config["model"]["n_head"],
            "pos_dim": config["model"]["pos_dim"],
            "move_avg": config["model"]["move_avg"],
            "teacher_temp": config["model"]["teacher_temp"]
        }

        file_path = "/gpfs01/berens/user/pgivehchian/ssl_neuron/outputs/results.csv"

        results_df = pd.DataFrame([results])

        if os.path.exists(file_path):
            results_df.to_csv(
                file_path,
                mode="a",
                header=False,
                index=False
            )
        else:
            results_df.to_csv(
                file_path,
                index=False
            )

        print(f"\nResults saved to {file_path}")

        return results_df



class BalancedSampler(Sampler):
    def __init__(self, labels):
        self.labels = np.asarray(labels)
        self.indices_by_label = {
            label: np.flatnonzero(self.labels == label)
            for label in np.unique(self.labels)
        }
        self.num_samples = max(
            len(indices) for indices in self.indices_by_label.values()
        )

    def __iter__(self):
        indices = []
        for class_indices in self.indices_by_label.values():
            indices.extend(class_indices.tolist())
            extra_count = self.num_samples - len(class_indices)
            if extra_count:
                draws = torch.randint(len(class_indices), (extra_count,))
                indices.extend(class_indices[draws].tolist())

        # Shuffle the balanced epoch.
        order = torch.randperm(len(indices))
        return iter([indices[i] for i in order])

    def __len__(self):
        return len(self.indices_by_label) * self.num_samples
