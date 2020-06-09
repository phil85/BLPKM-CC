from scipy.spatial.distance import cdist
import gurobipy as gb
import numpy as np


def update_centers(X, labels):
    unique_labels = np.unique(labels)
    centers = np.zeros((len(unique_labels), X.shape[1]))

    for i in unique_labels:
        idx = labels == i
        centers[i] = X[idx, :].mean(axis=0)
    return centers


def assign_objects(X, centers, must_link_pairs, cannot_link_pairs):
    m = gb.Model()

    object_ids = np.arange(X.shape[0])
    center_ids = np.arange(centers.shape[0])
    distances = cdist(X, centers)

    assignments = {(o, c): distances[o, c] for o in object_ids
                   for c in center_ids}
    x = m.addVars(assignments, obj=assignments, vtype=gb.GRB.BINARY)

    m.addConstrs(x.sum(i, '*') == 1 for i in object_ids)
    m.addConstrs(x[i, c] == x[j, c] for c in center_ids
                 for i, j in must_link_pairs)
    m.addConstrs(x[i, c] + x[j, c] <= 1 for c in center_ids
                 for i, j in cannot_link_pairs)
    m.addConstrs(x.sum('*', c) >= 1 for c in center_ids)
    m.optimize()

    cluster_ids = np.array([c for i, c in x.keys() if x[i, c].X > 0.5])
    return cluster_ids


def get_ofv(X, centers, cluster_ids):
    ofv = np.sqrt(((X - centers[cluster_ids, :]) ** 2).sum(axis=1)).sum()
    return ofv


def blpkm_cc(X, n_clusters, ml=[], cl=[], random_state=None, max_iter=100):
    # Choose initial cluster centers randomly

    np.random.seed(random_state)
    center_ids = np.random.choice(np.arange(X.shape[0]), size=n_clusters,
                                  replace=False)
    centers = X[center_ids, :]

    # Assign objects
    labels = assign_objects(X, centers, ml, cl)
    best_labels = labels

    # Update centers
    centers = update_centers(X, labels)

    # Compute ofv
    best_ofv = get_ofv(X, centers, labels)

    n_iter = 0
    while n_iter < max_iter:

        # Assign objects
        labels = assign_objects(X, centers, ml, cl)

        # Update centers
        centers = update_centers(X, labels)

        # Compute ofv
        current_ofv = get_ofv(X, centers, labels)

        # Check stopping criterion
        if current_ofv >= best_ofv:
            break
        else:
            best_labels = labels
            best_ofv = current_ofv

        # Increase iteration counter
        n_iter += 1

    return best_labels
