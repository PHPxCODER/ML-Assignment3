import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import random

# Function to calculate Euclidean distance between two points
def euclidean_distance(point_a, point_b):
    return np.sqrt(np.sum((point_a - point_b) ** 2))

# Function to initialize centroids using k-means++ method
def initialize_centroids(data, k):
    """Initialize centroids using k-means++ method"""
    centroids = []
    # Choose first centroid randomly
    first_centroid_idx = random.randint(0, len(data) - 1)
    centroids.append(data[first_centroid_idx])
    
    # Choose remaining centroids
    for i in range(1, k):
        # Calculate distances from points to nearest centroid
        min_distances = np.array([min([euclidean_distance(point, centroid) for centroid in centroids]) 
                                 for point in data])
        
        # Square distances to give more weight to distant points
        min_distances = min_distances ** 2
        
        # Choose next centroid with probability proportional to distance squared
        probabilities = min_distances / np.sum(min_distances)
        cumulative_probs = np.cumsum(probabilities)
        r = random.random()
        
        for j, p in enumerate(cumulative_probs):
            if r < p:
                centroids.append(data[j])
                break
    
    return np.array(centroids)

# Function to assign points to clusters
def assign_clusters(data, centroids):
    """Assign each point to the closest centroid"""
    clusters = []
    for point in data:
        distances = [euclidean_distance(point, centroid) for centroid in centroids]
        cluster_idx = np.argmin(distances)
        clusters.append(cluster_idx)
    return np.array(clusters)

# Function to update centroids
def update_centroids(data, clusters, k):
    """Update centroids based on mean of points in each cluster"""
    centroids = np.zeros((k, data.shape[1]))
    counts = np.zeros(k)
    
    for i, point in enumerate(data):
        cluster_idx = clusters[i]
        centroids[cluster_idx] += point
        counts[cluster_idx] += 1
    
    # Handle empty clusters by reinitializing
    for i in range(k):
        if counts[i] == 0:
            # If a cluster is empty, initialize with a random point
            centroids[i] = data[random.randint(0, len(data) - 1)]
        else:
            centroids[i] /= counts[i]
    
    return centroids

# Function to calculate cost (sum of squared distances)
def calculate_cost(data, clusters, centroids):
    """Calculate the cost as sum of squared distances to assigned centroids"""
    cost = 0
    for i, point in enumerate(data):
        centroid = centroids[clusters[i]]
        cost += euclidean_distance(point, centroid) ** 2
    return cost

# Function to normalize data
def normalize_data(data):
    """Normalize each feature to [0, 1] range"""
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    return (data - min_vals) / (max_vals - min_vals)

# Main K-means function
def k_means(data, k, max_iterations=100, convergence_threshold=0.0001):
    """Perform k-means clustering"""
    # Normalize the data
    normalized_data = normalize_data(data)
    
    # Initialize centroids
    centroids = initialize_centroids(normalized_data, k)
    prev_cost = float('inf')
    
    # Iterative process
    for iteration in range(max_iterations):
        # Assign points to clusters
        clusters = assign_clusters(normalized_data, centroids)
        
        # Update centroids
        centroids = update_centroids(normalized_data, clusters, k)
        
        # Calculate cost
        cost = calculate_cost(normalized_data, clusters, centroids)
        
        # Check for convergence
        if abs(prev_cost - cost) < convergence_threshold:
            print(f"Converged after {iteration + 1} iterations")
            break
        
        prev_cost = cost
    
    return clusters, centroids, normalized_data

# Function to plot clustering results
def plot_clusters(data, clusters, centroids, k, title):
    """Plot the clusters and centroids"""
    plt.figure(figsize=(10, 6))
    
    # Create a colormap
    colors = ListedColormap(['red', 'blue', 'green', 'purple', 'orange'])
    
    # Plot the data points with cluster colors
    plt.scatter(data[:, 0], data[:, 1], c=clusters, cmap=colors, s=50, alpha=0.8)
    
    # Plot centroids
    plt.scatter(
        centroids[:, 0] * (np.max(data[:, 0]) - np.min(data[:, 0])) + np.min(data[:, 0]),
        centroids[:, 1] * (np.max(data[:, 1]) - np.min(data[:, 1])) + np.min(data[:, 1]),
        s=200, marker='X', c=range(k), cmap=colors, edgecolors='black', linewidth=2
    )
    
    plt.title(title)
    plt.xlabel('Feature x1')
    plt.ylabel('Feature x2')
    plt.grid(alpha=0.3)
    plt.colorbar(label='Cluster')
    plt.savefig(f'kmeans_k{k}.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Load the dataset
    data = pd.read_csv('kmeans_blobs.csv')
    X = data.values
    
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Run k-means for k=2
    print("Running k-means for k=2")
    clusters_k2, centroids_k2, normalized_data = k_means(X, k=2)
    plot_clusters(X, clusters_k2, centroids_k2, k=2, title='K-means Clustering (k=2)')
    
    # Run k-means for k=3
    print("Running k-means for k=3")
    clusters_k3, centroids_k3, normalized_data = k_means(X, k=3)
    plot_clusters(X, clusters_k3, centroids_k3, k=3, title='K-means Clustering (k=3)')
    
    print("Clustering completed and plots saved.")

if __name__ == "__main__":
    main()