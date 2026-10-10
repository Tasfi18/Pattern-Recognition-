# -*- coding: utf-8 -*-
"""
Forgy & K-Means
---------------
Refactored version of the clustering code while preserving the main functionality.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load_data():
    return pd.read_csv(HERE / "clustering_data.csv", header=None)


def ask_for_ks(n_points, count=3):
    """Ask for `count` cluster sizes, each between 1 and the number of points."""
    ks = []
    while len(ks) < count:
        answer = input(f"Enter the number of clusters (k{len(ks) + 1}): ").strip()
        if answer.isdigit() and 1 <= int(answer) <= n_points:
            ks.append(int(answer))
        else:
            print(f"Please enter a whole number from 1 to {n_points}.")
    return ks

# =====================================================
# Helper Functions
# =====================================================
def random_seeds(k, df):
    """Randomly choose k seed points."""
    seeds = df.sample(n=k, random_state=42).reset_index(drop=True)
    return seeds

def compute_distances(points, df, k):
    """Compute distances from each point to the nearest centroid."""
    distances = []
    for i in range(df.shape[0]):
        dist_list = [
            (np.sqrt((df[0][i] - points[0][j])**2 + (df[1][i] - points[1][j])**2),
             (points[0][j], points[1][j]))
            for j in range(k)
        ]
        distances.append(min(dist_list))
    return distances

def calculate_cluster_sums(distances, df, k, seed_points):
    """Calculate sums of all matching points for each cluster."""
    sum_x, sum_y, count = np.zeros(k), np.zeros(k), np.zeros(k)
    for i in range(df.shape[0]):
        for j in range(k):
            if distances[i][1] == (seed_points[0][j], seed_points[1][j]):
                sum_x[j] += df[0][i]
                sum_y[j] += df[1][i]
                count[j] += 1
    return sum_x, sum_y, count

def update_seeds(sum_x, sum_y, count, k, old_seeds):
    """Update seed points based on cluster means.

    A cluster that lost all its points keeps its old centre instead of dividing by zero.
    """
    new_points = [
        [sum_x[i] / count[i], sum_y[i] / count[i]] if count[i] > 0
        else [old_seeds[0][i], old_seeds[1][i]]
        for i in range(k)
    ]
    new_seeds = pd.DataFrame(new_points, columns=[0, 1])
    return old_seeds, new_seeds

def WCSS(distances):
    """Within-Cluster Sum of Squares."""
    return sum(np.square(d[0]) for d in distances)

def plot_clusters(df, distances, centroids, k, title):
    """Plot dataset with cluster assignments and centroids."""
    # tab20 pairs a light and a dark shade of each colour; it repeats after 10 clusters
    palette = plt.get_cmap("tab20")
    colors = [palette((2 * j + 1) % 20) for j in range(k)]
    centroid_colors = [palette((2 * j) % 20) for j in range(k)]

    for i in range(df.shape[0]):
        for j in range(k):
            if distances[i][1] == (centroids[0][j], centroids[1][j]):
                plt.scatter(df[0][i], df[1][i], color=colors[j])

    for i in range(k):
        plt.scatter(centroids[0][i], centroids[1][i], color=centroid_colors[i], edgecolor='black', s=120)

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(title)
    plt.show()

# =====================================================
# Forgy Algorithm
# =====================================================
def forgy_algorithm(df, ks, plot=True):
    wcss_list = []
    for k in ks:
        seeds = random_seeds(k, df)
        distances = compute_distances(seeds, df, k)
        sum_x, sum_y, count = calculate_cluster_sums(distances, df, k, seeds)
        old_seeds, seeds = update_seeds(sum_x, sum_y, count, k, seeds)

        iterations = 1
        while not seeds.equals(old_seeds):
            distances = compute_distances(seeds, df, k)
            sum_x, sum_y, count = calculate_cluster_sums(distances, df, k, seeds)
            old_seeds, seeds = update_seeds(sum_x, sum_y, count, k, seeds)
            iterations += 1

        print(f"\n--- Forgy: k = {k} ---")
        print(f"Iterations to converge: {iterations-1}")
        print("Final centroids:")
        for i in range(k):
            print(f"Cluster {i+1}: ({float(seeds[0][i])}, {float(seeds[1][i])})")

        wcss = WCSS(distances)
        print(f"WCSS: {wcss}")
        wcss_list.append(wcss)

        if plot:
            plot_clusters(df, distances, seeds, k, f"Forgy Clustering (k={k})")
    return wcss_list

# =====================================================
# K-Means Algorithm
# =====================================================
def k_means_first_run(seed_points, df, k):
    clusters = [[float(seed_points[0][i]), float(seed_points[1][i])] for i in range(k)]
    cluster_counts = [0] * k

    # Count initial seeds
    for j in range(k):
        cluster_counts[j] = sum((seed_points[0][j] == df[0][i]) and (seed_points[1][j] == df[1][i])
                                for i in range(df.shape[0]))

    for i in range(df.shape[0]):
        dist_list = [
            (np.sqrt((df[0][i] - clusters[j][0])**2 + (df[1][i] - clusters[j][1])**2),
             (clusters[j][0], clusters[j][1]))
            for j in range(k)
        ]
        min_dist, cx, cy = min(dist_list)[0], min(dist_list)[1][0], min(dist_list)[1][1]

        if not ((df[0][i], df[1][i]) in zip(seed_points[0], seed_points[1])):
            for n in range(k):
                if (cx, cy) == (clusters[n][0], clusters[n][1]):
                    clusters[n][0] = (clusters[n][0] * cluster_counts[n] + df[0][i]) / (cluster_counts[n] + 1)
                    clusters[n][1] = (clusters[n][1] * cluster_counts[n] + df[1][i]) / (cluster_counts[n] + 1)
                    cluster_counts[n] += 1
    return clusters

def k_means_second_run(df, clusters, k):
    distances = []
    for i in range(df.shape[0]):
        dist_list = [
            (np.sqrt((df[0][i] - clusters[j][0])**2 + (df[1][i] - clusters[j][1])**2),
             (clusters[j][0], clusters[j][1]))
            for j in range(k)
        ]
        distances.append(min(dist_list))
    return distances

def k_means_algorithm(df, ks, plot=True):
    wcss_list = []
    for k in ks:
        seeds = random_seeds(k, df)
        clusters = k_means_first_run(seeds, df, k)
        distances = k_means_second_run(df, clusters, k)

        print(f"\n--- K-Means: k = {k} ---")
        print("Final centroids:")
        for i in range(k):
            print(f"Cluster {i+1}: ({clusters[i][0]}, {clusters[i][1]})")

        wcss = WCSS(distances)
        print(f"WCSS: {wcss}")
        wcss_list.append(wcss)

        centroids_df = pd.DataFrame(clusters, columns=[0, 1])
        if plot:
            plot_clusters(df, distances, centroids_df, k, f"K-Means Clustering (k={k})")
    return wcss_list

# =====================================================
# Run Algorithms & Compare
# =====================================================
if __name__ == "__main__":
    df = load_data()
    print("Dataset Shape:", df.shape)
    print(df.head())

    ks = ask_for_ks(df.shape[0])

    wcss_forgy = forgy_algorithm(df, ks)
    wcss_kmeans = k_means_algorithm(df, ks)

    # Plot WCSS vs k
    plt.plot(ks, wcss_forgy, label="Forgy")
    plt.plot(ks, wcss_kmeans, label="K-Means")
    plt.xlabel("k")
    plt.ylabel("WCSS")
    plt.title("WCSS vs k (Forgy vs K-Means)")
    plt.legend()
    plt.show()
