from scipy.spatial.distance import cdist
import gurobipy as gb
import numpy as np


def update_centers(X, centers, n_clusters, labels):
    """Update positions of cluster centers

    Args:
        X (np.array): feature vectors of objects
        centers (np.array): current positions of cluster centers
        n_clusters (int): predefined number of clusters
        labels (np.array): current cluster assignments of objects

    Returns:
        np.array: the updated positions of cluster centers

    """
    for i in range(n_clusters):
        centers[i] = X[labels == i, :].mean(axis=0)
    return centers


def assign_objects(X, centers, ml, cl):
    """Assigns objects to clusters

    Args:
        X (np.array): feature vectors of objects
        centers (np.array): current positions of cluster centers
        ml (list): must-link pairs as a list of tuples
        cl (list): cannot-link pairs as a list of tuples

    Returns:
        np.array: cluster labels for objects

    """

    # Compute model input
    n = X.shape[0]
    k = centers.shape[0]
    distances = cdist(X, centers)
    assignments = {(i, j): distances[i, j] for i in range(n) for j in range(k)}

    # Create model
    m = gb.Model()

    # Add binary decision variables
    y = m.addVars(assignments, obj=assignments, vtype=gb.GRB.BINARY)

    # Add constraints
    m.addConstrs(y.sum(i, '*') == 1 for i in range(n))
    m.addConstrs(y.sum('*', j) >= 1 for j in range(k))
    m.addConstrs(y[i, j] == y[i_, j] for j in range(k) for i, i_ in ml)
    m.addConstrs(y[i, j] + y[i_, j] <= 1 for j in range(k) for i, i_ in cl)

    # Determine optimal solution
    m.optimize()

    # Get labels from optimal assignment
    labels = np.array([j for i, j in y.keys() if y[i, j].X > 0.5])

    return labels


def get_total_distance(X, centers, labels):
    """Computes total distance between objects and cluster centers

    Args:
        X (np.array): feature vectors of objects
        centers (np.array): current positions of cluster centers
        labels (np.array): current cluster assignments of objects

    Returns:
        float: total distance

    """
    dist = np.sqrt(((X - centers[labels, :]) ** 2).sum(axis=1)).sum()
    return dist


def blpkm_cc(X, n_clusters, ml=[], cl=[], random_state=None, max_iter=100):
    """Finds partition of X subject to must-link and cannot-link constraints

    Args:
        X (np.array): feature vectors of objects
        n_clusters (int): predefined number of clusters
        ml (list): must-link pairs as a list of tuples
        cl (list): cannot-link pairs as a list of tuples
        random_state (int, RandomState instance): random state
        max_iter (int): maximum number of iterations of blpkm_cc algorithm

    Returns:
        np.array: cluster labels of objects

    """

    # Choose initial cluster centers randomly
    np.random.seed(random_state)
    center_ids = np.random.choice(np.arange(X.shape[0]), size=n_clusters,
                                  replace=False)
    centers = X[center_ids, :]

    # Assign objects
    labels = assign_objects(X, centers, ml, cl)

    # Initialize best labels
    best_labels = labels

    # Update centers
    centers = update_centers(X, centers, n_clusters, labels)

    # Compute total distance
    best_total_distance = get_total_distance(X, centers, labels)

    n_iter = 0
    while n_iter < max_iter:

        # Assign objects
        labels = assign_objects(X, centers, ml, cl)

        # Update centers
        centers = update_centers(X, centers, n_clusters, labels)

        # Compute total distance
        total_distance = get_total_distance(X, centers, labels)

        # Check stopping criterion
        if total_distance >= best_total_distance:
            break
        else:
            # Update best labels and best total distance
            best_labels = labels
            best_total_distance = total_distance

        # Increase iteration counter
        n_iter += 1

    return best_labels
