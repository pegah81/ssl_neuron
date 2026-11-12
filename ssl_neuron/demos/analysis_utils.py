import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os


def plot_skeletons(skeletons, titles=None, figsize=(12, 5), node_size=2):
    """
    Plot a list of Skeleton objects in XY and XZ views.
    """

    if not isinstance(skeletons, (list, tuple)):
        skeletons = [skeletons]

    n = len(skeletons)

    if n == 0:
        print("No skeletons to plot.")
        return

    fig, axes = plt.subplots(
        n, 2,
        figsize=(figsize[0], figsize[1] * n),
        squeeze=False
    )

    for i, skeleton in enumerate(skeletons):

        features = np.asarray(skeleton.features)
        neighbors = skeleton.neighbors

        # Assuming features columns are X, Y, Z
        x = features[:, 0]
        y = features[:, 1]
        z = features[:, 2]

        # Convert neighbor dictionary into unique edges
        edges = set()

        for node, adjacent_nodes in neighbors.items():
            for neighbor in adjacent_nodes:
                edge = tuple(sorted((int(node), int(neighbor))))
                edges.add(edge)

        # Plot XY and XZ
        for ax, coord1, coord2, xlabel, ylabel, view in [
            (axes[i, 0], x, y, "X", "Y", "XY view"),
            (axes[i, 1], x, z, "X", "Z", "XZ view")
        ]:

            for node1, node2 in edges:
                ax.plot(
                    [coord1[node1], coord1[node2]],
                    [coord2[node1], coord2[node2]],
                    color="black",
                    linewidth=0.5,
                    alpha=0.7
                )

            ax.scatter(coord1, coord2, s=node_size)

            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(view)
            ax.set_aspect("equal", adjustable="datalim")

            ax.invert_yaxis() if view == "XY view" else None

        title = titles[i] if titles is not None else f"Skeleton {i}"

        axes[i, 0].set_title(f"{title} — XY view")
        axes[i, 1].set_title(f"{title} — XZ view")

    plt.tight_layout()
    plt.show()




def plot_random_skeletons_by_type(
    cell_meta,
    skeletons,
    cell_types,
    n_samples=5,
    random_state=42,
    node_size=2
):
    """
    Randomly sample skeletons from each cell type and plot XY and XZ views.

    Each cell type gets its own figure:
        Row 1: XY views
        Row 2: XZ views
        Columns: number of sampled cells
    """

    rng = np.random.default_rng(random_state)

    for cell_type in cell_types:

        subset = cell_meta[
            cell_meta["celltype_final"] == cell_type
        ]

        available_ids = [
            str(cell_id)
            for cell_id in subset["cell_id"]
            if str(cell_id) in skeletons
        ]

        if len(available_ids) == 0:
            print(f"No skeletons available for {cell_type}")
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

            skeleton = skeletons[cell_id]
            features = np.asarray(skeleton.features)
            neighbors = skeleton.neighbors

            x = features[:, 0]
            y = features[:, 1]
            z = features[:, 2]

            edges = set()

            for node, adjacent_nodes in neighbors.items():
                for neighbor in adjacent_nodes:
                    edges.add(tuple(sorted((int(node), int(neighbor)))))

            ax = axes[0, j]

            for node1, node2 in edges:
                ax.plot(
                    [x[node1], x[node2]],
                    [y[node1], y[node2]],
                    color="black",
                    linewidth=0.5
                )

            ax.scatter(x, y, s=node_size)
            ax.set_title(f"{cell_id}\nXY view", fontsize=9)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_aspect("equal", adjustable="datalim")
            ax.invert_yaxis()

            ax = axes[1, j]

            for node1, node2 in edges:
                ax.plot(
                    [x[node1], x[node2]],
                    [z[node1], z[node2]],
                    color="black",
                    linewidth=0.5
                )

            ax.scatter(x, z, s=node_size)
            ax.set_title(f"{cell_id}\nXZ view", fontsize=9)
            ax.set_xlabel("X")
            ax.set_ylabel("Z")
            ax.set_aspect("equal", adjustable="datalim")

        fig.suptitle(
            f"{cell_type} — {n} randomly selected cells",
            fontsize=14
        )

        plt.tight_layout()
        plt.show()