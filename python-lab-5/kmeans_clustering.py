import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def initialize_centroids(K, data):
    centroids = []
    for _ in range(K):
        centroid = data[np.random.randint(0,np.size(data,0))]
        centroids.append(centroid)
    return np.array(centroids)

def assign_points_to_clusters(data, centroids):
    assignments = []

    for point in data:
        distance_to_centroids = []
        
        for centroid in centroids:
            distance = np.linalg.norm(point - centroid)
            distance_to_centroids.append(distance)
        
        assignment = np.argmin(distance_to_centroids)
        assignments.append(assignment)
    
    return assignments

def update_centroids(K, data, assignments):
    new_centroids = []
    
    for k in range(K):
        cluster = []
        for i in range(len(data)):
            if (assignments[i] == k):
                cluster.append(data[i]) 

        new_centroid = np.mean(cluster,0)
        new_centroids.append(new_centroid)

    return np.array(new_centroids)

def compute_error(data, labels, centroids):
    error = 0

    for i in range(np.size(data, 0)):
        centroid = centroids[labels[i]]
        error += np.linalg.norm(data[i] - centroid)**2

    return error

def kmeans_clustering(K, data, n_its, tol):

    data = np.array(data)
    centroids = initialize_centroids(K, data)
    d_error = 100
    error = [100]
    t = 0

    while (t < n_its) and (d_error > tol):
        t = t+1
        labels = assign_points_to_clusters(data, centroids)
        centroids = update_centroids(K, data, labels)
        error.append(compute_error(data, labels, centroids))
        d_error = np.abs((error[t-1] - error[t]) / error[t-1])
        print('Iteration:', t, end='\r')

    return labels, centroids , error[-1]






def plot_kmeans_scatter_2D(data, centers):
    plt.figure()
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis')
    plt.scatter(centers[:, 0], centers[:, 1], marker='X', s=200, c='red')
    plt.show()

def plot_kmeans_scatter_3D(data, centers):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter(data[:,0], data[:,1], data[:,2], marker='.', c=labels, alpha=1)
    ax.scatter(centers[:,0], centers[:,1], centers[:,2], marker='X', s=500, c='red', alpha=1)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$z$')
    ax.set_aspect('equal')
    # ax.legend(loc="lower right")
    fig.tight_layout()

def plot_kmeans_error(Ks, errors, best_K, path):
    fig = plt.figure()
    plt.plot(Ks, errors)
    plt.axvline(x=best_K, color='r', linestyle='--', label='Opt. K')
    plt.xlabel('$K$')
    plt.ylabel('WCSS')
    plt.legend(loc='upper right')
    fig.tight_layout()
    fig.savefig(path+r'\kmeans_error.png', dpi=300)

def kmeans_clustering(n_Ks, data):
    Ks = np.linspace(1, n_Ks, n_Ks)
    best_K = None
    errors = []
    best_silh_score = -1

    for K in Ks:

        K = int(K)
        kmeans = KMeans(n_clusters=K)
        kmeans.fit(data)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_
        error = kmeans.inertia_
        errors.append(error)

        if (K >= 2) and (K <= (data.shape[0]-1)):
            silh_score = silhouette_score(data, labels)

            if silh_score > best_silh_score:
                best_K = K
                best_labels = labels
                best_centroids = centers
                best_silh_score = silh_score
    
    print('Best K:', best_K)
    return best_labels, best_centroids, Ks, errors, best_K