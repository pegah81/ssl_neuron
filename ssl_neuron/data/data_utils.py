import numpy as np
import pandas as pd
import networkx as nx
from scipy.spatial.transform import Rotation as R
from ssl_neuron.utils import neighbors_to_adjacency, remap_neighbors
import torch
import pandas as pd
from tqdm import tqdm
import os
from sklearn.metrics.pairwise import cosine_distances
from collections import Counter
from sklearn.metrics import balanced_accuracy_score

from ssl_neuron.datasets import GraphDataset
from ssl_neuron.utils import neighbors_to_adjacency_torch, compute_eig_lapl_torch_batch
from ssl_neuron.graphdino import create_model

def connect_graph(adj_matrix, neighbors, features, verbose=False):
    """
    Check if graph consists of only one connected component. If not iterativly connect two points of two unconnected components with shortest distance between them.
    Args:
        adj_matrix: adjacency matrix of graph (N x N)
        neighbors: dict of neighbors per node
        features: features per node (N x D)
    """
    neighbors2 = {k: set(v) for k, v in neighbors.items()}
    G = nx.Graph(adj_matrix)
    num_comp = nx.number_connected_components(G)
    count = 1
    while num_comp > 1:
        components = {i: l for i, l in enumerate(list(nx.connected_components(G)))}
        components_ids = list(components.keys())
        for i, c_id in enumerate(components_ids):
            nodes = components[c_id]
            leaf_nodes = [n for n in nodes if len(neighbors2[n]) == 1]

            if len(leaf_nodes) > 0:

                min_comp_dist = np.inf
                min_comp_dist_id = -1
                min_comp_dist_node = -1
                for i, l in enumerate(leaf_nodes):
                    ne = neighbors2[l]
                    ne = list(ne)[0]

                    node_pos = features[l][:3]

                    nodes_pos_diff = ((features[:, :3] - node_pos) ** 2).sum(axis=1)
                    nodes_pos_diff[ne] = np.inf
                    nodes_pos_diff[l] = np.inf
                    nodes_pos_diff[list(nodes)] = np.inf

                    min_dist_id = np.argmin(nodes_pos_diff)
                    min_dist = np.min(nodes_pos_diff)

                    if min_comp_dist > min_dist:
                        min_comp_dist = min_dist
                        min_comp_dist_id = min_dist_id
                        min_comp_dist_node = l

                if min_comp_dist_id != -1 and min_comp_dist_node != -1:
                    neighbors2[min_comp_dist_id].add(min_comp_dist_node)
                    neighbors2[min_comp_dist_node].add(min_comp_dist_id)

        adj_matrix = neighbors_to_adjacency(neighbors2, range(len(neighbors2)))
        G = nx.Graph(adj_matrix)
        num_comp = nx.number_connected_components(G)
        if verbose:
            print(count, num_comp)
        count += 1
        
    return adj_matrix, neighbors2


def rotate_cell(cell_id, morphology, df):
    """ 
    Rotate neurons vertically with respect to pia.
    Args:
        cell_id: str
        morphology: AllenSDK morphology object
        df: pandas dataframe containing angles per neuron
    """
    z_rot = df[df['specimen_id']==cell_id]['upright_angle'].values[0]
    rot1 = R.from_euler('z', z_rot, degrees=True).as_matrix()
    rot_list = list(rot1.flatten()) + [0, 0, 0]
    morphology.apply_affine(rot_list)


    x_rot = df[df['specimen_id']==cell_id]['estimated_slice_angle'].values[0]
    if not pd.isna(x_rot):
        rot2 = R.from_euler('x', x_rot, degrees=True).as_matrix()
        rot_list2 = list(rot2.flatten()) + [0, 0, 0]
        morphology.apply_affine(rot_list2)

    return morphology


def remove_axon(neighbors, features, soma_id):
    """ Removes all nodes and edges in graph marked as axons.

        Feature dimensions:
            0 - 2: xyz coordinates
            3: radius
            4 - 7: One-hot encoding of compartment type:
                4: soma
                5: axon
                6: dendrite
                7: apical dendrite

        Args:
            neighbors: Dict of node id mapping to the node's neighbors.
            features: Node features (N x 8)
            soma id: Soma node index (int)

        Returns:
            neighbors: Updated neighbor dict without axon nodes.
            features: Updated feature array without axon nodes (M x 8).
            soma id: Updated soma node index.
    """
    # Get node indices corresponding to axon nodes.
    axon_mask = (features[:, 5] == 1)
    axon_idcs = list(np.where(axon_mask)[0])

    # Remove axon nodes from features.
    features = features[~axon_mask]

    # Remove axon nodes from neighbors.
    for key in axon_idcs:
        del neighbors[key]

    for key in neighbors:
        for n in list(neighbors[key]):
            if n in axon_idcs:
                neighbors[key].remove(n)

    # Re-map node indices to go from 0 .. M
    neighbors, old2new = remap_neighbors(neighbors)

    # Remap soma index.
    soma_id = old2new[soma_id]

    return neighbors, features, soma_id



def extract_latents(config):

    cell_meta = pd.read_csv(os.path.join(config['data']['path'], 'cell_meta.csv'))
    cell_meta['cell_id'] = cell_meta['cell_id'].astype(str)

    model = create_model(config)

    state_dict = torch.load(os.path.join(config['trainer']['ckpt_dir'], f"{config['trainer']['trial']}.pt"))

    model.load_state_dict(state_dict)

    model.eval()
    model.cuda();

    dset = GraphDataset(config, mode='all')


    latents = np.zeros((dset.num_samples, config['model']['dim']))

    cell_meta['latent'] = pd.Series(
        [None] * len(cell_meta),
        index=cell_meta.index,
        dtype=object
    )

    cell_meta['cell_id'] = cell_meta['cell_id'].astype(str)

    for i in tqdm(range(dset.num_samples)):
        feat, neigh, cell_id = dset.__getsingleitem__(i)
        cell_id = str(cell_id)

        adj = neighbors_to_adjacency_torch(
            neigh, list(neigh.keys())
        ).float().cuda()[None, :]

        lapl = compute_eig_lapl_torch_batch(
            adj,
            pos_enc_dim=config['model']['pos_dim']
        ).float().cuda()

        feat = torch.from_numpy(feat).float().cuda()[None, :]

        with torch.no_grad():
            embedding = model.student_encoder.forward(feat, adj, lapl)[0]
            embedding = embedding.cpu().numpy().squeeze()

        latents[i] = embedding

        mask = cell_meta['cell_id'] == cell_id

        for idx in cell_meta.index[mask]:
            cell_meta.at[idx, 'latent'] = embedding

    print("\nEmbedding extraction complete!")
    print("Latents matrix shape:", latents.shape)
    print("Rows with embeddings:", cell_meta['latent'].notna().sum())

    cell_meta.to_csv(
        os.path.join(config['trainer']['latents_dir'], f"{config['trainer']['trial']}.csv"),
        index=False
    )


def predict_knn_accuracy(model, train_loader, val_loader, device, k=10):

    model.eval()

    def extract(loader):
        X, y = [], []

        with torch.no_grad():
            for data in loader:
                f1, f2, a1, a2, cell_ids, labels = data

                f1 = f1.float().to(device)
                a1 = a1.float().to(device)

                lapl = compute_eig_lapl_torch_batch(a1)

                z = model.student_encoder(f1, a1, lapl)[0]

                X.append(z.cpu().numpy())
                y.extend(labels)

        return np.concatenate(X), np.asarray(y)

    X_train, y_train = extract(train_loader)
    X_val, y_val = extract(val_loader)

    def predict(X_query, X_reference, y_reference):
        distances = cosine_distances(X_query, X_reference)
        indices = np.argsort(distances, axis=1)[:, :k]

        return np.array([
            Counter(y_reference[n]).most_common(1)[0][0]
            for n in indices
        ])

    # Leave-one-out training accuracy
    train_distances = cosine_distances(X_train, X_train)
    np.fill_diagonal(train_distances, np.inf)

    train_indices = np.argsort(train_distances, axis=1)[:, :k]

    train_predictions = np.array([
        # the most common one in neighbours is the prediction
        Counter(y_train[n]).most_common(1)[0][0]
        for n in train_indices
    ])

    train_acc = balanced_accuracy_score(y_train, train_predictions)

    # Validation predictions using training embeddings
    val_predictions = predict(X_val, X_train, y_train)

    val_acc = balanced_accuracy_score(y_val, val_predictions)

    model.train()

    return train_acc, val_acc


def save_knn_accuracy(epoch, train_acc, val_acc, config):

    path = os.path.join(
        config["trainer"]["losses_dir"],
        config["trainer"]["trial"],
        "knn_accuracy.csv"
    )

    os.makedirs(os.path.dirname(path), exist_ok=True)

    pd.DataFrame([{
        "epoch": epoch,
        "train_knn_accuracy": train_acc,
        "val_knn_accuracy": val_acc
    }]).to_csv(
        path,
        mode="a",
        header=not os.path.exists(path),
        index=False
    )